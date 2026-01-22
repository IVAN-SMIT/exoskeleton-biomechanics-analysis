# exoskeleton-biomechanics-analysis
Проект по имитационному моделированию взаимодействия человека и экзоскелета нижних конечностей с использованием данных оптического захвата движения (motion capture) в среде MuJoCo

# Biomechanical Analysis of Lower-Limb Exoskeleton Interaction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-2.3+-green.svg)](https://mujoco.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Проект по имитационному моделированию взаимодействия человека и экзоскелета нижних конечностей с использованием данных оптического захвата движения (motion capture) в среде MuJoCo.

## Цель проекта

Разработка инструмента для моделирования и оценки биомеханического взаимодействия человека и экзоскелета нижних конечностей в динамических условиях на основе реальных данных motion capture.

## Основные возможности

- **Визуализация движения человека**: Импорт и 3D-визуализация данных motion capture (Vicon, Qualisys, Plug-in Gait)
- **Параметрическое моделирование экзоскелета**: Иерархическая модель нижних конечностей с физически обоснованными ограничениями
- **Интеграция систем**: Объединение модели человека и экзоскелета в единую динамическую систему MuJoCo
- **Биомеханический анализ**: Количественная оценка кинематических, динамических и энергетических параметров
- **Оптимизация параметров**: Исследование влияния жесткости привязки на эффективность и комфорт

## Ключевые метрики анализа

- **Кинематические**: Траектории маркеров, углы суставов, RMS-отклонения
- **Динамические**: Внутрисуставные моменты, контактные силы, усилия в ремнях
- **Энергетические**: Механическая работа, пиковая и средняя мощность
- **Интегральные**: Нагрузка на позвоночник (L5/S1 compression)

## Быстрый старт

### Установка зависимостей

```bash
# Клонирование репозитория
git clone https://github.com/your-username/exoskeleton-biomechanics-analysis.git
cd exoskeleton-biomechanics-analysis

# Установка зависимостей
pip install -r requirements.txt

# Или через conda
conda env create -f environment.yml
conda activate exoskeleton-analysis
```

Запуск симуляции
```bash

# Базовый запуск с данными из папки data/
python src/main.py ./data/raw/

# С указанием силы привязки (0.0-1.0)
python src/main.py ./data/raw/ --strength=0.7

# Для генерации отчетов и графиков
python src/analysis/generate_report.py --input ./data/raw/ --output ./outputs/
```

Авторы:

Смирнов Иван @IVAN_SMIT
Нагорный Леонид @DRAMMATICPENNY59
Гусаров Сергей @ser4jk
