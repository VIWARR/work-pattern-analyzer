from typing import Optional

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.utils import resample


class ClusteringFeatureBuilder:
    """Подготовка признаков для кластеризации"""
    def __init__(self, features: list[str]):
        self.features = features

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.features].replace([np.inf, -np.inf], np.nan)
        missing_ration = X.isna().mean().mean()
        if missing_ration > 0.2:
            raise ValueError(f"Слишком много пропущенных значений в признаках для кластеризации: {missing_ration:.2%}")
        return X.fillna(0)
    

class OptimalKSelector:
    """
    Решает проблему неопределенности количества кластеров (Unsupervised Learning).
    
    Использует гибридный подход:
    1. Метод локтя для оценки компактности.
    2. Коэффициент силуэта для оценки разделимости.
    
    Алгоритм ищет 'точку насыщения', где добавление нового кластера не дает 
    значимого прироста объясненной дисперсии (min_inertia_gain).
    """
    def __init__(
            self,   
            k_range: range = range(2, 6),
            silhouette_tolerance: float = 0.05,
            min_inertia_gain: float = 0.25):
        self.k_range = k_range
        self.silhouette_tolerance = silhouette_tolerance
        self.min_inertia_gain = min_inertia_gain

    def select(self, X: pd.DataFrame, random_state: int = 42) -> int:
        metrics = self._preprocesed_k(X, random_state)
        metrics = metrics.sort_values(by='k').reset_index(drop=True)
        best_k = metrics.loc[metrics['silhouette'].idxmax(), 'k']
        best_sil = metrics['silhouette'].max()

        for i in range(len(metrics) - 1):
            row = metrics.iloc[i]
            next_row = metrics.iloc[i + 1]
            sil_drop = best_sil - next_row['silhouette']
            inertia_gain = (row['inertia'] - next_row['inertia']) / row['inertia']
            
            if sil_drop <= self.silhouette_tolerance and inertia_gain >= self.min_inertia_gain:
                best_k = next_row['k']

        return int(best_k)


    def _preprocesed_k(self, X: pd.DataFrame, random_state: int) -> pd.DataFrame:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        results = []
        for k in self.k_range:
            model = KMeans(n_clusters=k, random_state=random_state, n_init=20)
            labels = model.fit_predict(X_scaled)
            results.append({
                'k': k,
                'inertia': model.inertia_,
                'silhouette': silhouette_score(X_scaled, labels)
            })

        return pd.DataFrame(results)
    

class KMeansClusterer:
    """
    Инкапсулирует Pipeline обучения модели K-Means.
    """
    def __init__ (self, n_clusters: int, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.pipeline: Optional[Pipeline] = None

    def fit_predict(self, X: pd.DataFrame) -> pd.DataFrame:
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('kmeans', KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=20))
        ])
        return self.pipeline.fit_predict(X)
    

class ClusterProfiler:
    """
    Восстанавливает физический смысл признаков после масштабирования.
    
    Рассчитывает 'типичного представителя' (центроид) каждого кластера 
    в исходных единицах измерения (часы, проценты). Это позволяет 
    аналитику интерпретировать каждый кластер как конкретный режим работы
    """
    def __init__(self, features_names: list[str]):
        self.feature_names = features_names

    def build(self, pipeline: Pipeline) -> pd.DataFrame:
        scaler = pipeline.named_steps['scaler']
        kmeans = pipeline.named_steps['kmeans']
        centroids = scaler.inverse_transform(kmeans.cluster_centers_)
        profiles = pd.DataFrame(centroids, columns=self.feature_names)
        profiles['cluster'] = profiles.index
        return profiles.round(3)


class BusinessLabeler:
    """
    Транслирует математические результаты кластеризации в бизнес-термины.
    
    Использует взвешенную Z-score оценку ключевых метрик (focus_density, session_depth) 
    для ранжирования кластеров от 'Пика продуктивности' до 'Операционки'.
    
    Метод позволяет автоматизировать маркировку новых данных без ручного анализа профилей.
    """
    def __init__(self, feature_weights: dict[str, float] | None = None):
        self.feature_weights = feature_weights or {'focus_density': 0.5, 'session_depth': 0.5}

    def label(self, profiles: pd.DataFrame) -> dict[int, str]:
        df = profiles.copy()
        for col in self.feature_weights:
            std = df[col].std(ddof=0)
            if std == 0 or np.isnan(std):
                df[col + '_z'] = 0.0
            else:
                df[col + '_z'] = (df[col] - df[col].mean()) / std

        df['score'] = sum(df[f"{col}_z"] * w for col, w in self.feature_weights.items())

        best_cluster = df.loc[df['score'].idxmax(), 'cluster']
        worst_cluster = df.loc[df['score'].idxmin(), 'cluster']
        labels = {}

        for _, row in df.iterrows():
            cid = int(row['cluster'])
            if cid == best_cluster:
                label = 'Пик продуктивности'
            elif cid == worst_cluster:
                label = 'Операционка и паузы'
            elif row['score'] > 0:
                label = 'Высокая интенсивность'
            elif row['score'] < 0:
                label = 'Поддерживающие задачи'
            else:
                label = 'Сбалансированный режим'

            labels[cid] = label
        
        return labels


class ClusteringStability:
    """
    Валидирует надежность выделенных паттернов поведения.
    
    Использует метод ресэмплинга (bootstrap) для проверки того, насколько 
    выделенные кластеры устойчивы к изменению выборки. Низкий скор стабильности 
    сигнализирует о 'размытости' паттернов в данных или избыточном числе K.
    """
    def score(self, X: pd.DataFrame, n_clusters: int, n_iter: int = 20) -> float:
        scores = []
        for i in range(n_iter):
            X_resampled = resample(X, random_state=42 + i)
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('kmeans', KMeans(
                    n_clusters=n_clusters,
                    n_init=10,
                    random_state=42 + i
                ))
            ])
            labels = pipeline.fit_predict(X_resampled)
            score = silhouette_score(
                pipeline.named_steps['scaler'].transform(X_resampled),
                labels
            )
            scores.append(score)
            
        return float(np.mean(scores))