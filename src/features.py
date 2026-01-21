import pandas as pd
import numpy as np
from src.config import FEATURE_COLUMNS_REQUIRED, PROCESSED_COLUMNS, COLUMN_MAPPING, WEIGHTS, logger

class DataPreprocesor:
    """
    Трансформатор для нормализации сырых данных.
    
    Выполняет приведение типов, временную синхронизацию и вставку пропущенных значений. 
    Гарантирует соответствие результирующего DataFrame контракту нисходящей обработки, 
    сохраняя неизменяемость исходного источника данных.
    """
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._standardize_columns(df)
        df = self._convert_dtypes(df)
        df = self._handle_missing_values(df)
        df = self._normalize_text(df)
        return df

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(columns=self.COLUMN_MAPPING)
    
    def _convert_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        for col in ['start_time', 'end_time']:
            delta = pd.to_timedelta(df[col].astype(str), errors='coerce')
            df[col] = df['date'] + delta

        df['duration_hours'] = pd.to_numeric(df['duration_hours'], errors='coerce')
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.dropna(subset=PROCESSED_COLUMNS)
        removed = before - len(df)

        if removed:
            logger.warning(f"Удалено {removed} строк с пропущенными значениями")
        return df
    
    def _normalize_text(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ['description', 'activity_type', 'week']:
            df[col] = df[col].str.strip().str.lower()
        return df
    

class ActivityClassifier:
    """
    Классификатор активностей для семантического анализа рабочего времени.

    Выполняет маппинг сырых типов активностей (activity_type) в укрупненные 
    аналитические категории на основе 
    словаря соответствий. 

    Обеспечивает консистентность данных для Feature Engineering и позволяет 
    отслеживать неразмеченные типы задач через механизм логирования пропусков.
    """
    def __init__(self, category_map: dict):
        self.category_map = category_map

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy
        df['activity_category'] = 'other'
        
        for category, value in self.category_map.items():
            mask = df['activity_type'].isin(value)
            df.loc[mask, 'activity_category'] = category

        self._log_unmatched(df)
        return df
    
    def _log_unmatched(self, df: pd.DataFrame) -> None:
        all_defined = set().union(*self.category_map.values())
        unmatched = set(df['activity_type'].inique()) - all_defined

        if unmatched:
            logger.warning(f"Активности категории Other: {unmatched}")


class FeatureEngineer:
    """
    Инженер признаков для кластерного анализа рабочих сессий.

    Вычисляет ключевые метрики продуктивности на основе временных интервалов 
    и категорий активностей. 
    Обеспечивает подготовку данных для алгоритмов кластеризации, 
    фокусируясь на глубине сессий и плотности концентрации.
    """
    def __init__(self, flow_category: str = 'deep_work',
                 main_flow_duration: float = 1.0,
                 overtime_threshold: int = 19,
                 productivity_weights: dict[str, float]= WEIGHTS):
        self.flow_category = flow_category
        self.main_flow_duration = main_flow_duration
        self.overtime_threshold = overtime_threshold
        self.productivity_weights = productivity_weights

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(df)
        df = df.copy().sort_values(['date', 'start_time']).reset_index(drop=True)
        df = self._create_time_features(df)
        df = self._create_behavior_features(df)
        df = self._create_sequence_features(df)
        df = self._create_composite_metrics(df)
        return df

    def _validate_input(self, df: pd.DataFrame) -> None:
        missing = FEATURE_COLUMNS_REQUIRED - set(df.columns)
        if missing:
            raise ValueError(f"Отсутствуют необходимые столбцы: {missing}")
        
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['hour_start'] = df['start_time'].dt.hour
        df['hour_end'] = df['end_time'].dt.hour
        df['time_start'] = df['start_time'].dt.hour + df['start_time'].dt.minute / 60
        df['time_end'] = df['end_time'].dt.hour + df['end_time'].dt.minute / 60
        df['is_over_time'] = (df['hour_end'] >= self.overtime_threshold).astype(int)
        df['month'] = df['date'].dt.month

        extracted_weeks = df['week'].str.extract(r'w(\d+)')
        df['week_num'] = pd.to_numeric(extracted_weeks[0], errors='coerce').fillna(0).astype(int)
        return df
    
    def _create_behavior_features(self, df: pd.DataFrame) -> pd.DataFrame:
        flow_condition = (
            (df['duration_hours'] >= self.min_flow_duration) &
            (df['activity_category'] == self.flow_category)
        )
        df['is_flow_session'] = flow_condition.astype(int)
        df['makers_hours'] = np.where(flow_condition, df['duration_hours'], 0)
        return df

    def _create_sequence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Сдвиги для анализа последовательностей
        df['prev_activity_type'] = df['activity_type'].shift(1)
        df['prev_end_time'] = df['end_time'].shift(1)
        df['prev_date'] = df['date'].shift(1)

        # Переключения контекста
        same_day = df['date'] == df['prev_date']
        different_activity = df['activity_type'] != df['prev_activity_type']
        df['is_context_switch'] = (same_day & different_activity).astype(int)

        # Время между задачами
        df['gap_minutes'] = np.where(
            same_day,
            (df['start_time'] - df['prev_end_time']).dt.total_seconds() / 60,
            0
        )

        # Очистка временных колонок
        df = df.drop(['prev_activity_type', 'prev_end_time', 'prev_date'], axis=1)
        return df

    def _create_composite_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        df['productivity_score'] = (
            df['activity_category'].map(self.productivity_weights)
            .fillna(0)
            * df['duration_hours']
        )
        return df