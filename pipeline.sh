#!/bin/bash

# Прерывать выполнение при любой ошибке
set -e

echo "Установка зависимостей..."
pip3 install -r requirements.txt

echo "Шаг 1: Генерация данных..."
python3 data_creation.py

if [ $? -ne 0 ]; then
    echo "Ошибка при создании данных."
    exit 1
fi


echo "Шаг 2: Предобработка данных..."
python3 model_preprocessing.py

if [ $? -ne 0 ]; then
    echo "Ошибка на этапе предобработки."
    exit 1
fi


echo "Шаг 3: Обучение модели..."
python3 model_preparation.py

if [ $? -ne 0 ]; then
    echo "Ошибка при обучении модели."
    exit 1
fi


echo "Шаг 4: Тестирование модели..."
python3 model_testing.py

if [ $? -ne 0 ]; then
    echo "Ошибка при тестировании модели."
    exit 1
fi


echo "Пайплайн успешно выполнен."