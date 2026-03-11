# Прогноз температуры
Учебный проект: генерация синтетических данных о температуре, предобработка, обучение линейной регрессии и проверка качества модели

## Что делает проект
- генерирует 20 CSV-датасетов (`data_creation.py`): температура с сезонным колебанием + гауссовский шум, 70% train и 30% test;
- масштабирует признаки `day` и `temperature` (`model_preprocessing.py`), сохраняет `scaler.pkl`;
- обучает `LinearRegression` по признаку `day` (`model_preparation.py`), сохраняет `model.pkl`;
- считает метрики `MSE` и `R2` на тестовых данных (`model_testing.py`).

## Быстрый запуск
```bash
chmod +x pipeline.sh
./pipeline.sh
```

## Структура
- `data/train`, `data/test` — исходные датасеты;
- `data/processed/train_scaled.csv` — подготовленные train-данные;
- `model.pkl`, `scaler.pkl` — обученная модель и scaler.
