import pandas as pd
from typing import List, Optional

from src.config import logger
from src.data_loader import DataLoader
from src.features import DataPreprocesor
from src.features import ActivityClassifier, FeatureEngineer, TimeAggregator
from src.clustering import (
    ClusteringFeatureBuilder, 
    OptimalKSelector, 
    KMeansClusterer, 
    ClusterProfiler, 
    BusinessLabeler, 
    ClusteringStability
)


class ProductivityPipeline:
    """
    Сквозной процесс генерации аналитического датасета (Feature Store).
    
    Управляет цепочкой: Загрузка -> Препроцессинг -> Классификация задач -> Инженерия признаков.
    Результатом является Daily Stats — фундаментальный уровень данных для всех 
    последующих ML-экспериментов.
    """
    def __init__(
      self,
      loader: DataLoader,
      preprocesor: DataPreprocesor,
      classifier: ActivityClassifier,
      engineer: FeatureEngineer,
      aggregator: TimeAggregator):

      self.loader = loader
      self.preprocesor = preprocesor
      self.classifier = classifier
      self.engineer = engineer
      self.aggregator = aggregator

    def run(self) -> pd.DataFrame:
        daily_stats = self.aggregator.aggregate_daily(
            self._build_features(self.loader.load())
        )
        return daily_stats

    def run_feature_data(self) -> pd.DataFrame:
        return self._build_features(self.loader.load())

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        processed = self.preprocesor.process(df)
        classified = self.classifier.classify(processed)
        featured = self.engineer.engineer_features(classified)
        return featured


class ClusteringPipeline:
    """
    Оркестратор холодного запуска и обучения модели сегментации.
    
    Связывает воедино подготовку признаков, автоматический подбор гиперпараметров (K) 
    и финальную бизнес-интерпретацию. Реализует паттерн 'End-to-End ML Pipeline', 
    минимизируя ручное вмешательство в процесс сегментации пользователей.
    """
    def __init__(
        self,
        cluster_features: list[str],
        k_selector: OptimalKSelector,
        labeler: BusinessLabeler,
        random_state: int = 42
    ):
        self.feature_builder = ClusteringFeatureBuilder(cluster_features)
        self.k_selector = k_selector
        self.labeler = labeler
        self.random_state = random_state
        self.clusterer: KMeansClusterer | None = None

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self.feature_builder.transform(df)

        n_clusters = self.k_selector.select(X)
        logger.info(f"Количество кластеров: {n_clusters}")

        self.clusterer = KMeansClusterer(n_clusters, self.random_state)
        cluster_ids = self.clusterer.fit_predict(X)

        result = df.copy()
        result['cluster'] = cluster_ids

        profiler = ClusterProfiler(self.feature_builder.features)
        profiles = profiler.build(self.clusterer.pipeline)

        label_map = self.labeler.label(profiles)
        result['cluster_label'] = result['cluster'].map(label_map)

        try:
            stability = ClusteringStability().score(X, n_clusters)
            logger.info(f"Стабильность кластеров: {stability:.3f}")
        except Exception as e:
            logger.warning(f"Проверка пропущена: {e}")

        return result