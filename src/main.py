import os
import sys
import numpy as np
from scipy.interpolate import interp1d
import scipy.io
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import time
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
_EPS = 1e-6  # Увеличили точность
SCALE = 0.001  # мм → метры

# Конфигурация сегментов с параметрами привязки
SEGMENT_CONFIG = [
    # Формат: (маркер1, маркер2, цвет_RGBA, радиус, жесткость_привязки)
    ('LASI', 'RASI', [0.2, 0.8, 0.2, 1], 0.015, 0.8),  # Таз
    # Правая нога
    ('RASI', 'RTHI', [0.6, 0.1, 0.1, 1], 0.015, 0.7),
    ('RTHI', 'RKNE', [0.8, 0.2, 0.2, 1], 0.015, 0.7),
    ('RKNE', 'RANK', [0.8, 0.2, 0.2, 1], 0.015, 0.7),
    ('RANK', 'RTOE', [0.9, 0.4, 0.4, 1], 0.012, 0.6),
    # Левая нога
    ('LASI', 'LTHI', [0.1, 0.1, 0.6, 1], 0.015, 0.7),
    ('LTHI', 'LKNE', [0.2, 0.2, 0.8, 1], 0.015, 0.7),
    ('LKNE', 'LANK', [0.2, 0.2, 0.8, 1], 0.015, 0.7),
    ('LANK', 'LTOE', [0.4, 0.4, 0.9, 1], 0.012, 0.6),
]

class RobustCylinder:
    """Улучшенный класс цилиндра с защитой от исчезновения"""
    def __init__(self, m1, m2, rgba, radius, attachment_strength):
        self.marker1 = m1
        self.marker2 = m2
        self.rgba = rgba
        self.radius = radius
        self.attachment_strength = attachment_strength
        # Текущие концы цилиндра
        self.end1 = np.zeros(3, dtype=np.float64)
        self.end2 = np.zeros(3, dtype=np.float64)
        # Минимальная и максимальная длина
        self.min_length = 0.02  # Минимальная длина цилиндра (2 см)
        self.max_length = 1.0   # Максимальная длина цилиндра (1 м)
        # История позиций для сглаживания
        self.history_len = 5
        self.position_history = []
        # Флаги состояния
        self.is_valid = True
        self.last_valid_end1 = np.zeros(3)
        self.last_valid_end2 = np.zeros(3, dtype=np.float64)
        self.last_valid_end2[:] = [0, 0, 0.1]  # Начальное смещение
        # Счетчик проблем
        self.problem_count = 0
        self.max_problems = 10

    def _validate_vector(self, vec, name="вектор"):
        """Проверка валидности вектора"""
        if np.any(np.isnan(vec)) or np.any(np.isinf(vec)):
            return False
        return True

    def _ensure_minimum_distance(self, p1, p2):
        """Гарантировать минимальное расстояние между точками"""
        vec = p2 - p1
        dist = np.linalg.norm(vec)
        if dist < self.min_length:
            # Если точки слишком близко, раздвигаем их
            if dist > _EPS:
                direction = vec / dist
            else:
                # Если расстояние почти нулевое, используем направление вверх
                direction = np.array([0.0, 0.0, 1.0])
            # Смещаем точки от центра
            center = (p1 + p2) / 2
            p1 = center - direction * (self.min_length / 2)
            p2 = center + direction * (self.min_length / 2)
            self.problem_count += 1
        return p1, p2

    def _smooth_update(self, current, target, strength, dt):
        """Плавное обновление позиции"""
        # Проверяем валидность данных
        if not self._validate_vector(current, "current") or not self._validate_vector(target, "target"):
            return current
        # Интерполяция с учетом времени
        alpha = min(1.0, strength * dt * 120.0)  # Нормализуем к частоте 120 Гц
        new_pos = current * (1 - alpha) + target * alpha
        return new_pos

    def update(self, marker1_pos, marker2_pos, dt):
        """Обновить позицию цилиндра с защитой от исчезновения"""
        # Проверяем валидность входных данных
        if not (self._validate_vector(marker1_pos, f"маркер {self.marker1}") and 
                self._validate_vector(marker2_pos, f"маркер {self.marker2}")):
            # Используем последние валидные позиции
            if self.is_valid:
                marker1_pos = self.last_valid_end1
                marker2_pos = self.last_valid_end2
            else:
                return  # Пропускаем обновление
        # Гарантируем минимальное расстояние между маркерами
        marker1_pos, marker2_pos = self._ensure_minimum_distance(marker1_pos, marker2_pos)
        # Плавное обновление позиций концов
        if len(self.position_history) == 0:
            # Первая инициализация
            self.end1 = marker1_pos.copy()
            self.end2 = marker2_pos.copy()
            self.last_valid_end1 = self.end1.copy()
            self.last_valid_end2 = self.end2.copy()
        else:
            # Плавное движение к целевым позициям
            self.end1 = self._smooth_update(self.end1, marker1_pos, self.attachment_strength, dt)
            self.end2 = self._smooth_update(self.end2, marker2_pos, self.attachment_strength, dt)
        # Гарантируем минимальное расстояние между концами цилиндра
        self.end1, self.end2 = self._ensure_minimum_distance(self.end1, self.end2)
        # Ограничиваем максимальную длину
        vec = self.end2 - self.end1
        dist = np.linalg.norm(vec)
        if dist > self.max_length:
            direction = vec / dist
            center = (self.end1 + self.end2) / 2
            self.end1 = center - direction * (self.max_length / 2)
            self.end2 = center + direction * (self.max_length / 2)
        # Сохраняем историю для сглаживания
        self.position_history.append((self.end1.copy(), self.end2.copy()))
        if len(self.position_history) > self.history_len:
            self.position_history.pop(0)
        # Применяем сглаживание по истории
        if len(self.position_history) > 1:
            avg_end1 = np.mean([p[0] for p in self.position_history], axis=0)
            avg_end2 = np.mean([p[1] for p in self.position_history], axis=0)
            # Легкое сглаживание
            blend_factor = 0.2
            self.end1 = self.end1 * (1 - blend_factor) + avg_end1 * blend_factor
            self.end2 = self.end2 * (1 - blend_factor) + avg_end2 * blend_factor
        # Сохраняем последние валидные позиции
        self.last_valid_end1 = self.end1.copy()
        self.last_valid_end2 = self.end2.copy()
        self.is_valid = True

    def get_endpoints(self):
        """Получить текущие концы цилиндра с гарантией минимальной длины"""
        # Дополнительная проверка перед возвратом
        vec = self.end2 - self.end1
        dist = np.linalg.norm(vec)
        if dist < self.min_length:
            # Корректируем на лету
            if dist > _EPS:
                direction = vec / dist
            else:
                direction = np.array([0.0, 0.0, 1.0])
            center = (self.end1 + self.end2) / 2
            end1_corrected = center - direction * (self.min_length / 2)
            end2_corrected = center + direction * (self.min_length / 2)
            return end1_corrected, end2_corrected
        return self.end1.copy(), self.end2.copy()

    def get_deviation(self, marker1_pos, marker2_pos):
        """Получить отклонение от маркеров"""
        if not (self._validate_vector(marker1_pos) and self._validate_vector(marker2_pos)):
            return 0.0
        dev1 = np.linalg.norm(self.end1 - marker1_pos)
        dev2 = np.linalg.norm(self.end2 - marker2_pos)
        # Проверка на NaN/Inf
        if np.isnan(dev1) or np.isinf(dev1) or np.isnan(dev2) or np.isinf(dev2):
            return 0.0
        return (dev1 + dev2) / 2

    def get_status(self):
        """Получить статус цилиндра"""
        vec = self.end2 - self.end1
        dist = np.linalg.norm(vec)
        return {
            'valid': self.is_valid,
            'length': dist,
            'problems': self.problem_count,
            'min_length_ok': dist >= self.min_length
        }

class DeviationMonitor:
    """Мониторинг отклонений маркеров от примитивов"""
    def __init__(self, segment_names, output_dir):
        self.segment_names = segment_names
        self.output_dir = output_dir
        self.deviation_history = {name: [] for name in segment_names}
        self.time_history = []
        # Статистика
        self.max_deviations = {name: 0 for name in segment_names}
        self.avg_deviations = {name: 0 for name in segment_names}
        self.deviation_counts = {name: 0 for name in segment_names}
        print(f"📊 Мониторинг отклонений инициализирован")
        print(f"   Графики будут сохранены в: {output_dir}")

    def update_deviations(self, time_val, deviations):
        """Обновить данные отклонений"""
        for name, dev in deviations.items():
            if name in self.deviation_history:
                # Проверяем на валидность
                if not np.isnan(dev) and not np.isinf(dev):
                    self.deviation_history[name].append(dev)
                    # Обновляем статистику
                    self.max_deviations[name] = max(self.max_deviations[name], dev)
                    self.avg_deviations[name] = (self.avg_deviations[name] * self.deviation_counts[name] + dev) / (self.deviation_counts[name] + 1)
                    self.deviation_counts[name] += 1
        self.time_history.append(time_val)

    def save_plots_and_data(self, attachment_strength, duration, cylinder_statuses=None):
        """Сохранить графики и данные в файлы"""
        if not self.time_history:
            print("⚠ Нет данных для сохранения графиков")
            return
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        # 1. Сохраняем сырые данные в CSV
        csv_filename = os.path.join(self.output_dir, f"deviation_data_{timestamp}.csv")
        self._save_csv_data(csv_filename)
        # 2. Создаем и сохраняем графики
        self._create_and_save_plots(timestamp, attachment_strength, duration)
        # 3. Сохраняем сводную статистику
        stats_filename = os.path.join(self.output_dir, f"deviation_stats_{timestamp}.txt")
        self._save_statistics(stats_filename, attachment_strength, duration, cylinder_statuses)
        print(f"✅ Данные и графики сохранены в папку: {self.output_dir}")
        print(f"   📄 Данные: deviation_data_{timestamp}.csv")
        print(f"   📈 Графики: deviation_plots_{timestamp}.png")
        print(f"   📊 Статистика: deviation_stats_{timestamp}.txt")

    def _save_csv_data(self, filename):
        """Сохранить данные в CSV файл"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                # Заголовок
                f.write("time_s," + ",".join(self.segment_names) + "\n")
                # Данные (отклонения в сантиметрах)
                min_len = min(len(self.time_history), 
                            min(len(h) for h in self.deviation_history.values()))
                for i in range(min_len):
                    f.write(f"{self.time_history[i]:.3f},")
                    deviations = [self.deviation_history[name][i] * 100 if i < len(self.deviation_history[name]) else 0 
                                for name in self.segment_names]
                    f.write(",".join(f"{d:.4f}" for d in deviations) + "\n")
            print(f"   ✅ CSV данные сохранены: {os.path.basename(filename)}")
        except Exception as e:
            print(f"   ❌ Ошибка сохранения CSV: {e}")

    def _create_and_save_plots(self, timestamp, attachment_strength, duration):
        """Создать и сохранить графики"""
        try:
            # Создаем фигуру с 4 субплогами
            fig = plt.figure(figsize=(16, 12))
            fig.suptitle(f'Отклонения маркеров от примитивов\nСила привязки: {attachment_strength:.1%}, Длительность: {duration:.1f}с', 
                        fontsize=16, fontweight='bold')
            # 1. График всех отклонений
            ax1 = plt.subplot(2, 2, 1)
            ax1.set_title('Отклонения по всем сегментам', fontsize=14)
            ax1.set_xlabel('Время (с)', fontsize=12)
            ax1.set_ylabel('Отклонение (см)', fontsize=12)
            ax1.grid(True, alpha=0.3)
            colors = plt.cm.Set3(np.linspace(0, 1, len(self.segment_names)))
            for i, name in enumerate(self.segment_names):
                if self.deviation_history[name]:
                    deviations_cm = [d * 100 for d in self.deviation_history[name]]
                    time_data = self.time_history[:len(deviations_cm)]
                    ax1.plot(time_data, deviations_cm, label=name, color=colors[i], linewidth=1.5, alpha=0.8)
            ax1.legend(loc='upper right', fontsize=9, ncol=2)
            # 2. График среднего и максимального отклонения
            ax2 = plt.subplot(2, 2, 2)
            ax2.set_title('Сводные показатели отклонений', fontsize=14)
            ax2.set_xlabel('Время (с)', fontsize=12)
            ax2.set_ylabel('Отклонение (см)', fontsize=12)
            ax2.grid(True, alpha=0.3)
            if self.time_history and all(self.deviation_history.values()):
                min_len = min(len(h) for h in self.deviation_history.values())
                avg_deviations = []
                max_deviations = []
                for i in range(min_len):
                    devs_at_time = [self.deviation_history[name][i] * 100
                                  for name in self.segment_names 
                                  if i < len(self.deviation_history[name])]
                    if devs_at_time:
                        avg_deviations.append(np.mean(devs_at_time))
                        max_deviations.append(np.max(devs_at_time))
                if avg_deviations:
                    time_slice = self.time_history[:min_len]
                    ax2.plot(time_slice, avg_deviations, 'b-', linewidth=2, label='Среднее отклонение', alpha=0.8)
                    ax2.plot(time_slice, max_deviations, 'r-', linewidth=2, label='Максимальное отклонение', alpha=0.8)
            ax2.legend(fontsize=10)
            # 3. Гистограмма средних отклонений по сегментам
            ax3 = plt.subplot(2, 2, 3)
            ax3.set_title('Средние отклонения по сегментам', fontsize=14)
            ax3.set_xlabel('Сегмент', fontsize=12)
            ax3.set_ylabel('Среднее отклонение (см)', fontsize=12)
            ax3.grid(True, alpha=0.3, axis='y')
            avg_deviations_cm = []
            for name in self.segment_names:
                if self.deviation_history[name]:
                    avg_dev = np.mean([d * 100 for d in self.deviation_history[name]])
                    avg_deviations_cm.append(avg_dev)
                else:
                    avg_deviations_cm.append(0)
            bars = ax3.bar(range(len(self.segment_names)), avg_deviations_cm, 
                          color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
            # Добавляем значения над столбцами
            for i, (bar, val) in enumerate(zip(bars, avg_deviations_cm)):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=9, rotation=0)
            ax3.set_xticks(range(len(self.segment_names)))
            ax3.set_xticklabels(self.segment_names, rotation=45, ha='right', fontsize=10)
            # 4. Box plot отклонений
            ax4 = plt.subplot(2, 2, 4)
            ax4.set_title('Распределение отклонений', fontsize=14)
            ax4.set_xlabel('Сегмент', fontsize=12)
            ax4.set_ylabel('Отклонение (см)', fontsize=12)
            ax4.grid(True, alpha=0.3, axis='y')
            # Подготавливаем данные для box plot
            box_data = []
            box_labels = []
            for name in self.segment_names:
                if self.deviation_history[name] and len(self.deviation_history[name]) > 0:
                    deviations_cm = [d * 100 for d in self.deviation_history[name]]
                    box_data.append(deviations_cm)
                    box_labels.append(name)
            if box_data:
                bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True)
                # Раскрашиваем box plot
                for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                ax4.set_xticklabels(box_labels, rotation=45, ha='right', fontsize=10)
            # Настраиваем макет
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            # Сохраняем графики
            plot_filename = os.path.join(self.output_dir, f"deviation_plots_{timestamp}.png")
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"   ✅ Графики сохранены: {os.path.basename(plot_filename)}")
        except Exception as e:
            print(f"   ❌ Ошибка создания графиков: {e}")
            import traceback
            traceback.print_exc()

    def _save_statistics(self, filename, attachment_strength, duration, cylinder_statuses=None):
        """Сохранить статистику в текстовый файл"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write("СТАТИСТИКА ОТКЛОНЕНИЙ МАРКЕРОВ ОТ ПРИМИТИВОВ\n")
                f.write("=" * 70 + "\n")
                f.write(f"Дата анализа: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Сила привязки: {attachment_strength:.1%}\n")
                f.write(f"Длительность анимации: {duration:.1f} с\n")
                f.write(f"Количество сегментов: {len(self.segment_names)}\n")
                f.write(f"Количество точек данных: {len(self.time_history)}\n")
                # Статус цилиндров
                if cylinder_statuses:
                    f.write("-" * 70 + "\n")
                    f.write("СТАТУС ЦИЛИНДРОВ:\n")
                    f.write("-" * 70 + "\n")
                    for name, status in cylinder_statuses.items():
                        f.write(f"{name:12s}: ")
                        if status['valid']:
                            f.write(f"✓ Валиден, длина={status['length']:.3f} м, ")
                            f.write(f"проблем={status['problems']}, ")
                            f.write(f"min_length_ok={'✓' if status['min_length_ok'] else '✗'}\n")
                        else:
                            f.write("✗ Невалиден\n")
                    f.write("\n")
                f.write("-" * 70 + "\n")
                f.write("СТАТИСТИКА ПО СЕГМЕНТАМ (в сантиметрах):\n")
                f.write("-" * 70 + "\n")
                # Общая статистика
                all_deviations = []
                for name in self.segment_names:
                    if self.deviation_history[name]:
                        all_deviations.extend([d * 100 for d in self.deviation_history[name]])
                if all_deviations:
                    f.write("ОБЩАЯ СТАТИСТИКА:\n")
                    f.write(f"  Среднее отклонение: {np.mean(all_deviations):.3f} см\n")
                    f.write(f"  Максимальное отклонение: {np.max(all_deviations):.3f} см\n")
                    f.write(f"  Минимальное отклонение: {np.min(all_deviations):.3f} см\n")
                    f.write(f"  Стандартное отклонение: {np.std(all_deviations):.3f} см\n")
                    f.write(f"  Медиана: {np.median(all_deviations):.3f} см\n")
                # Статистика по каждому сегменту
                f.write("ПО СЕГМЕНТАМ:\n")
                f.write("Сегмент\t\tСреднее\t\tМаксимум\tТочек\tСтатус\n")
                f.write("-" * 70 + "\n")
                for name in self.segment_names:
                    if self.deviation_history[name]:
                        deviations_cm = [d * 100 for d in self.deviation_history[name]]
                        avg = np.mean(deviations_cm)
                        max_dev = np.max(deviations_cm)
                        count = len(deviations_cm)
                        # Определяем статус по величине отклонения
                        if avg < 1.0:
                            status = "✓ Отлично"
                        elif avg < 3.0:
                            status = "○ Нормально"
                        elif avg < 5.0:
                            status = "⚠ Высокое"
                        else:
                            status = "✗ Критическое"
                        f.write(f"{name:12s}\t{avg:8.3f}\t{max_dev:8.3f}\t{count:6d}\t{status}\n")
                f.write("\n" + "=" * 70 + "\n")
                f.write("КОНЕЦ ОТЧЕТА\n")
                f.write("=" * 70 + "\n")
            print(f"   ✅ Статистика сохранена: {os.path.basename(filename)}")
        except Exception as e:
            print(f"   ❌ Ошибка сохранения статистики: {e}")


def load_simple_marker(filepath):
    """Загрузка маркерных данных из .mat файла"""
    try:
        name = os.path.basename(filepath).replace('.mat','')
        mat = scipy.io.loadmat(filepath)
        candidates = []
        for k, v in mat.items():
            if k.startswith('__'): continue
            if isinstance(v, np.ndarray):
                arr = np.asarray(v)
                if arr.ndim == 2 and arr.shape[1] == 3:
                    candidates.append((k,arr))
                elif arr.ndim == 1 and arr.size % 3 == 0:
                    N = arr.size // 3
                    candidates.append((k, arr.reshape(N,3)))
        if not candidates: 
            return {}
        name, xyz = candidates[0]
        t = np.arange(xyz.shape[0]) / 120.0
        arr = np.column_stack([t, xyz*SCALE])
        return {name:arr}
    except Exception as e:
        print(f"⚠ Ошибка загрузки {filepath}: {e}")
        return {}

def create_interpolators(markers):
    """Создание интерполяторов для маркеров"""
    funcs = {}
    for name, arr in markers.items():
        if arr.shape[0] > 1:
            t, xyz = arr[:,0], arr[:,1:4]
            # Проверяем данные на NaN
            if np.any(np.isnan(xyz)):
                print(f"⚠ В данных маркера {name} обнаружены NaN значения")
                # Заменяем NaN на ближайшие валидные значения
                for i in range(3):
                    nan_mask = np.isnan(xyz[:, i])
                    if np.any(nan_mask):
                        # Интерполируем NaN значения
                        not_nan_mask = ~nan_mask
                        if np.any(not_nan_mask):
                            xyz[nan_mask, i] = np.interp(
                                t[nan_mask], t[not_nan_mask], xyz[not_nan_mask, i]
                            )
                        else:
                            xyz[:, i] = 0  # Если все значения NaN
            funcs[name] = interp1d(t, xyz, axis=0, bounds_error=False,
                                   fill_value=(xyz[0],xyz[-1]), assume_sorted=True)
        else:
            funcs[name] = lambda t_val, v=arr[0,1:4]: v
    return funcs

def robust_vector_to_quaternion(v_from, v_to):
    """Создание кватерниона вращения с защитой от ошибок"""
    # Нормализуем векторы
    norm_from = np.linalg.norm(v_from)
    norm_to = np.linalg.norm(v_to)
    if norm_from < _EPS or norm_to < _EPS:
        # Если векторы нулевые, возвращаем единичный кватернион
        return np.array([1.0, 0.0, 0.0, 0.0])
    v_from = v_from / norm_from
    v_to = v_to / norm_to
    # Проверяем на NaN/Inf
    if np.any(np.isnan(v_from)) or np.any(np.isinf(v_from)) or \
       np.any(np.isnan(v_to)) or np.any(np.isinf(v_to)):
        return np.array([1.0, 0.0, 0.0, 0.0])
    dot = np.dot(v_from, v_to)
    dot = np.clip(dot, -1.0, 1.0)  # Защита от ошибок округления
    if dot > 0.999999:
        # Векторы почти совпадают
        return np.array([1.0, 0.0, 0.0, 0.0])
    elif dot < -0.999999:
        # Векторы противоположны
        # Находим ортогональную ось
        if abs(v_from[0]) < 0.9:
            axis = np.cross(v_from, np.array([1.0, 0.0, 0.0]))
        else:
            axis = np.cross(v_from, np.array([0.0, 1.0, 0.0]))
        axis_norm = np.linalg.norm(axis)
        if axis_norm < _EPS:
            return np.array([0.0, 0.0, 0.0, 1.0])
        axis = axis / axis_norm
        return np.array([0.0, axis[0], axis[1], axis[2]])
    angle = np.arccos(dot)
    axis = np.cross(v_from, v_to)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < _EPS:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axis / axis_norm
    half_angle = angle / 2
    s = np.sin(half_angle)
    return np.array([np.cos(half_angle), axis[0]*s, axis[1]*s, axis[2]*s])

def init_custom_geoms(viewer, max_geoms=1000):
    """Инициализация пользовательской геометрии с увеличенным лимитом"""
    viewer.user_scn.ngeom = max_geoms
    for i in range(max_geoms):
        viewer.user_scn.geoms[i].type = mujoco.mjtGeom.mjGEOM_SPHERE
        viewer.user_scn.geoms[i].size[:] = [0.0, 0.0, 0.0]
        viewer.user_scn.geoms[i].rgba[:] = [0.0, 0.0, 0.0, 0.0]
    viewer.user_scn.ngeom = 0

def create_robust_cylinder(p1, p2, radius, rgba, viewer, geom_index):
    """Создание цилиндра с защитой от ошибок"""
    # Проверяем валидность точек
    if (np.any(np.isnan(p1)) or np.any(np.isinf(p1)) or 
        np.any(np.isnan(p2)) or np.any(np.isinf(p2))):
        return geom_index
    vec = p2 - p1
    dist = np.linalg.norm(vec)
    # Минимальная длина для отображения
    min_visible_length = 0.001  # 1 мм
    if dist < min_visible_length:
        # Если цилиндр слишком короткий, все равно отображаем его
        # но с минимальной видимой длиной
        if dist > _EPS:
            direction = vec / dist
        else:
            direction = np.array([0.0, 0.0, 1.0])
        # Смещаем точки чтобы создать видимый цилиндр
        center = (p1 + p2) / 2
        p1 = center - direction * (min_visible_length / 2)
        p2 = center + direction * (min_visible_length / 2)
        dist = min_visible_length
    # Центр цилиндра
    center_pos = (p1 + p2) / 2
    # Создаем кватернион вращения
    z_axis = np.array([0.0, 0.0, 1.0])
    if dist > _EPS:
        direction = vec / dist
        quat = robust_vector_to_quaternion(z_axis, direction)
    else:
        quat = np.array([1.0, 0.0, 0.0, 0.0])
    # Конвертируем кватернион в матрицу
    mat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, quat)
    # Полувысота цилиндра
    half_length = max(dist / 2, 0.0005)  # Минимум 0.5 мм
    # Создаем цилиндр
    try:
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[geom_index],
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[radius, half_length, 0],
            pos=center_pos,
            mat=mat,
            rgba=rgba
        )
        return geom_index + 1
    except Exception as e:
        print(f"⚠ Ошибка создания цилиндра: {e}")
        return geom_index

# ================================
# НОВЫЕ ФУНКЦИИ ДЛЯ ГРАФИКОВ АНАЛИЗА
# ================================

def calculate_segment_angles(marker_positions):
    """Расчет углов суставов на основе маркеров"""
    angles = {}
    # Правый тазобедренный
    if all(m in marker_positions for m in ['RASI', 'RKNE']):
        thigh = marker_positions['RKNE'] - marker_positions['RASI']
        vertical = np.array([0, 0, 1])
        norm_thigh = np.linalg.norm(thigh)
        if norm_thigh > _EPS:
            angle = np.arccos(np.dot(thigh, vertical) / (norm_thigh * 1.0))
            angles['right_hip'] = angle
    # Правый коленный
    if all(m in marker_positions for m in ['RTHI', 'RKNE', 'RANK']):
        thigh = marker_positions['RKNE'] - marker_positions['RTHI']
        shank = marker_positions['RANK'] - marker_positions['RKNE']
        norm_thigh = np.linalg.norm(thigh)
        norm_shank = np.linalg.norm(shank)
        if norm_thigh > _EPS and norm_shank > _EPS:
            cos_angle = np.dot(thigh, shank) / (norm_thigh * norm_shank)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.pi - np.arccos(cos_angle)
            angles['right_knee'] = angle
    # Левый тазобедренный
    if all(m in marker_positions for m in ['LASI', 'LKNE']):
        thigh = marker_positions['LKNE'] - marker_positions['LASI']
        vertical = np.array([0, 0, 1])
        norm_thigh = np.linalg.norm(thigh)
        if norm_thigh > _EPS:
            angle = np.arccos(np.dot(thigh, vertical) / (norm_thigh * 1.0))
            angles['left_hip'] = angle
    # Левый коленный
    if all(m in marker_positions for m in ['LTHI', 'LKNE', 'LANK']):
        thigh = marker_positions['LKNE'] - marker_positions['LTHI']
        shank = marker_positions['LANK'] - marker_positions['LKNE']
        norm_thigh = np.linalg.norm(thigh)
        norm_shank = np.linalg.norm(shank)
        if norm_thigh > _EPS and norm_shank > _EPS:
            cos_angle = np.dot(thigh, shank) / (norm_thigh * norm_shank)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.pi - np.arccos(cos_angle)
            angles['left_knee'] = angle
    return angles

def save_biomechanical_plots(time_history, joint_angles_history, rms_history, output_dir, attachment_strength, duration):
    """Сохранение графиков в стиле скриншота: углы, RMS, отклонения"""
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle(f'Биомеханический анализ: gait_analysis\nСила привязки: {attachment_strength:.1%}, Длительность: {duration:.1f}с',
                     fontsize=16, fontweight='bold')

        # 1. Углы суставов
        ax1 = plt.subplot(2, 3, 1)
        joint_names = ['right_hip', 'right_knee', 'left_hip', 'left_knee']
        colors = ['r', 'orange', 'b', 'cyan']
        has_data = False
        for i, joint in enumerate(joint_names):
            if joint in joint_angles_history and len(joint_angles_history[joint]) > 0:
                angles_deg = np.degrees(joint_angles_history[joint])
                time_slice = time_history[:len(angles_deg)]
                ax1.plot(time_slice, angles_deg, color=colors[i], label=joint.replace('_', ' ').title(), linewidth=2)
                has_data = True
        if has_data:
            ax1.set_xlabel('Время (с)')
            ax1.set_ylabel('Угол (град)')
            ax1.set_title('Углы суставов')
            ax1.grid(True, alpha=0.3)
            ax1.legend()

        # 2. RMS отклонение
        ax2 = plt.subplot(2, 3, 2)
        if rms_history:
            ax2.plot(time_history[:len(rms_history)], np.array(rms_history) * 100, 'g-', linewidth=2)
            ax2.set_xlabel('Время (с)')
            ax2.set_ylabel('RMS отклонение (см)')
            ax2.set_title('RMS отклонение маркеров')
            ax2.grid(True, alpha=0.3)

        # 3. Разница углов (право vs лево) — бедро
        ax4 = plt.subplot(2, 3, 4)
        if 'right_hip' in joint_angles_history and 'left_hip' in joint_angles_history:
            rh = np.array(joint_angles_history['right_hip'][:min(len(joint_angles_history['right_hip']), len(joint_angles_history['left_hip']))])
            lh = np.array(joint_angles_history['left_hip'][:len(rh)])
            diff = np.degrees(rh - lh)
            ax4.plot(time_history[:len(diff)], diff, 'm-', linewidth=1.5)
            ax4.set_xlabel('Время (с)')
            ax4.set_ylabel('ΔУгол (град)')
            ax4.set_title('Разница: правый - левый бедро')
            ax4.grid(True, alpha=0.3)

        # 4. Разница углов — колено
        ax5 = plt.subplot(2, 3, 5)
        if 'right_knee' in joint_angles_history and 'left_knee' in joint_angles_history:
            rk = np.array(joint_angles_history['right_knee'][:min(len(joint_angles_history['right_knee']), len(joint_angles_history['left_knee']))])
            lk = np.array(joint_angles_history['left_knee'][:len(rk)])
            diff = np.degrees(rk - lk)
            ax5.plot(time_history[:len(diff)], diff, 'c-', linewidth=1.5)
            ax5.set_xlabel('Время (с)')
            ax5.set_ylabel('ΔУгол (град)')
            ax5.set_title('Разница: правое - левое колено')
            ax5.grid(True, alpha=0.3)

        # 5. Сводная статистика
        ax6 = plt.subplot(2, 3, 6)
        metrics = []
        values = []
        if rms_history:
            metrics.append('Ср. RMS')
            values.append(np.mean(rms_history) * 100)
            metrics.append('Макс RMS')
            values.append(np.max(rms_history) * 100)
        for joint in joint_names:
            if joint in joint_angles_history:
                angles = joint_angles_history[joint]
                if angles:
                    metrics.append(f'Макс {joint}')
                    values.append(np.max(np.degrees(angles)))
        if metrics:
            bars = ax6.bar(range(len(metrics)), values, color=plt.cm.tab10(np.linspace(0,1,len(metrics))))
            ax6.set_xticks(range(len(metrics)))
            ax6.set_xticklabels(metrics, rotation=45, ha='right', fontsize=9)
            ax6.set_ylabel('Значение')
            ax6.set_title('Сводные метрики')
            ax6.grid(True, alpha=0.3, axis='y')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_path = os.path.join(output_dir, f"biomechanical_analysis_plots_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Графики биомеханического анализа сохранены: {os.path.basename(plot_path)}")
    except Exception as e:
        print(f"⚠ Ошибка при создании графиков анализа: {e}")
        import traceback
        traceback.print_exc()


def visualize_and_monitor(markers, interp_funcs, model, data, viewer, t_max, attachment_strength, output_dir):
    """Визуализация с мониторингом отклонений + запись углов и RMS для графиков"""
    # Создаем улучшенные цилиндры
    robust_cylinders = []
    segment_names = []
    print("🔧 Создание цилиндров с защитой от исчезновения...")
    for m1, m2, rgba, radius, strength in SEGMENT_CONFIG:
        if m1 in interp_funcs and m2 in interp_funcs:
            cylinder = RobustCylinder(m1, m2, rgba, radius, strength)
            robust_cylinders.append(cylinder)
            segment_name = f"{m1}-{m2}"
            segment_names.append(segment_name)
            print(f"   ✅ Цилиндр {segment_name}: привязка {strength:.1%}, мин.длина={cylinder.min_length*100:.1f}см")
        else:
            print(f"   ⚠ Пропущен {m1}-{m2}: маркеры не найдены")
    if not robust_cylinders:
        print("❌ Нет цилиндров для отображения!")
        return

    # Инициализация истории для биомеханического анализа
    time_history = []
    joint_angles_history = {'right_hip': [], 'right_knee': [], 'left_hip': [], 'left_knee': []}
    rms_history = []

    # Создаем монитор отклонений (как в оригинале)
    monitor = DeviationMonitor(segment_names, output_dir)
    joint_points = set()
    for cylinder in robust_cylinders:
        joint_points.add(cylinder.marker1)
        joint_points.add(cylinder.marker2)

    t = 0
    dt = 1/120.0
    prev_time = 0
    frame_count = 0
    max_geoms_used = 0

    try:
        while viewer.is_running() and t <= t_max:
            viewer.user_scn.ngeom = 0
            geom_index = 0
            frame_count += 1

            marker_positions = {}
            for name, func in interp_funcs.items():
                pos = func(t)
                if np.any(np.isnan(pos)) or np.any(np.isinf(pos)):
                    pos = np.zeros(3)
                marker_positions[name] = pos

            # === ЗАПИСЬ БИОМЕХАНИЧЕСКИХ ДАННЫХ ===
            time_history.append(t)
            angles = calculate_segment_angles(marker_positions)
            for joint in joint_angles_history:
                joint_angles_history[joint].append(angles.get(joint, np.nan))

            # Считаем текущий RMS отклонения
            deviations_list = []
            for cylinder in robust_cylinders:
                m1_pos = marker_positions.get(cylinder.marker1, np.zeros(3))
                m2_pos = marker_positions.get(cylinder.marker2, np.zeros(3))
                dev = cylinder.get_deviation(m1_pos, m2_pos)
                if not (np.isnan(dev) or np.isinf(dev)):
                    deviations_list.append(dev)
            if deviations_list:
                current_rms = np.sqrt(np.mean(np.square(deviations_list)))
                rms_history.append(current_rms)
            else:
                rms_history.append(0.0)

            # === ВИЗУАЛИЗАЦИЯ (оригинальная логика) ===
            for name, pos in marker_positions.items():
                if geom_index >= 990:
                    break
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[geom_index],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.02, 0, 0],
                    pos=pos,
                    mat=np.eye(3).flatten(),
                    rgba=[1, 0, 0, 1]
                )
                geom_index += 1

            current_time = t
            time_step = current_time - prev_time if prev_time > 0 else dt
            prev_time = current_time
            deviations = {}
            for i, cylinder in enumerate(robust_cylinders):
                marker1_pos = marker_positions.get(cylinder.marker1, np.zeros(3))
                marker2_pos = marker_positions.get(cylinder.marker2, np.zeros(3))
                cylinder.update(marker1_pos, marker2_pos, time_step)
                cyl_end1, cyl_end2 = cylinder.get_endpoints()
                if geom_index < 990:
                    geom_index = create_robust_cylinder(
                        cyl_end1, cyl_end2, cylinder.radius, cylinder.rgba, viewer, geom_index
                    )
                deviation = cylinder.get_deviation(marker1_pos, marker2_pos)
                segment_name = segment_names[i]
                deviations[segment_name] = deviation

            if deviations:
                monitor.update_deviations(t, deviations)

            for joint_name in joint_points:
                if joint_name in interp_funcs and geom_index < 990:
                    p = interp_funcs[joint_name](t)
                    rgba = [0, 1, 0, 0.7]
                    size = 0.015
                    if joint_name in ['RANK', 'LANK', 'RTOE', 'LTOE']:
                        size = 0.012
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[geom_index],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[size, 0, 0],
                        pos=p,
                        mat=np.eye(3).flatten(),
                        rgba=rgba
                    )
                    geom_index += 1

            max_geoms_used = max(max_geoms_used, geom_index)
            viewer.user_scn.ngeom = geom_index
            viewer.sync()
            t += dt

    except KeyboardInterrupt:
        print("\n⚠ Симуляция прервана пользователем (Ctrl+C)")
    except Exception as e:
        print(f"\n⚠ Ошибка в симуляции: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Сохраняем данные DeviationMonitor (как в оригинале)
        final_statuses = {}
        for i, cylinder in enumerate(robust_cylinders):
            segment_name = segment_names[i]
            final_statuses[segment_name] = cylinder.get_status()
        monitor.save_plots_and_data(attachment_strength, t, final_statuses)

        # === ДОПОЛНИТЕЛЬНО: сохраняем графики биомеханического анализа ===
        print("💾 Сохранение графиков углов и RMS...")
        save_biomechanical_plots(
            time_history,
            joint_angles_history,
            rms_history,
            output_dir,
            attachment_strength,
            t
        )


def main():
    if len(sys.argv) < 2:
        print("Использование: python script.py <путь_к_папке> [--strength=N]")
        print("  --strength=N: сила привязки (0.0-1.0, по умолчанию 0.7)")
        print("    1.0 = жесткая привязка, 0.0 = полная свобода")
        return
    data_dir = sys.argv[1]
    attachment_strength = 0.7
    for arg in sys.argv[2:]:
        if arg.startswith('--strength='):
            try:
                attachment_strength = float(arg.split('=')[1])
                attachment_strength = max(0.0, min(1.0, attachment_strength))
            except:
                print(f"Ошибка парсинга аргумента: {arg}")
    if not os.path.isdir(data_dir):
        print("❌ Папка не найдена")
        return

    # Загрузка маркеров
    print("📂 Загрузка маркеров из .mat файлов...")
    markers = {}
    t_max = 0
    mat_files = []
    for f in os.listdir(data_dir):
        if f.lower().endswith('.mat'):
            mat_files.append(f)
            res = load_simple_marker(os.path.join(data_dir, f))
            markers.update(res)
            for arr in res.values():
                t_max = max(t_max, arr[-1, 0])
    if not markers:
        print("❌ Нет маркеров для загрузки")
        return
    marker_names = list(markers.keys())
    print(f"✅ Загружено {len(marker_names)} маркеров из {len(mat_files)} файлов")
    print(f"   Маркеры: {', '.join(sorted(marker_names))}")

    # Центрирование
    print("🎯 Центрирование данных...")
    center = np.mean([arr[0, 1:4] for arr in markers.values()], axis=0)
    for arr in markers.values():
        arr[:, 1:4] -= center

    # Создание интерполяторов
    print("📊 Создание интерполяторов...")
    interp_funcs = create_interpolators(markers)

    # Простая модель для визуализации
    xml = '''<mujoco model="exoskeleton">
  <worldbody>
    <camera name="cam" pos="0 0 1.5" euler="-30 0 0"/>
    <light pos="0 0 3" dir="0 0 -1" diffuse="1 1 1"/>
  </worldbody>
</mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    print(f"\n🎬 Запуск визуализации")
    print(f"   Сила привязки: {attachment_strength:.1%}")
    print(f"   Длительность: {t_max:.1f} секунд")
    print(f"   Результаты сохраняются в: {data_dir}")

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            init_custom_geoms(viewer, max_geoms=1000)
            visualize_and_monitor(
                markers, interp_funcs, model, data, viewer, t_max,
                attachment_strength, data_dir
            )
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("🏁 Программа завершена")
    print("="*60)


if __name__ == "__main__":
    main()
