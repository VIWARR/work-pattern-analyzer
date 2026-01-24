from src.config import ProjectConfig, setup_logging, CATEGORY_MAP, CLUSTER_FEATURES, FEATURE_WEIGHTS
from src.data_loader import DataLoader
from src.features import DataPreprocesor, ActivityClassifier, FeatureEngineer, TimeAggregator
from src.clustering import OptimalKSelector, BusinessLabeler
from src.pipeline import ProductivityPipeline, ClusteringPipeline
from app import run_dashboard
import pandas as pd

def main():
    # 1. Настройка окружения
    setup_logging()
    config = ProjectConfig()

    # 2. Инициализация пайплайнов
    product_pipeline = ProductivityPipeline(
        DataLoader(config),
        DataPreprocesor(),
        ActivityClassifier(CATEGORY_MAP),
        FeatureEngineer(),
        TimeAggregator()
    )

    clustering = ClusteringPipeline(
        k_selector=OptimalKSelector(
            k_range=range(2, 6),
            silhouette_tolerance=0.05,
            min_inertia_gain=0.30
        ),
        cluster_features=CLUSTER_FEATURES,
        labeler=BusinessLabeler(FEATURE_WEIGHTS)
    )

    # 3. Расчеты
    print(">>> Этап 1. Дата инжениринг и расчет дневных показателей")
    feature_df = product_pipeline.run_feature_data()
    daily_stats = product_pipeline.run()

    print(">>> Этап 2. Кластеризация")
    cluster_df = clustering.run(daily_stats)
    
    # 4. Визуализация
    print(">>> Этап 3. Подготовка и запуск дашборда")
    run_dashboard(clust_df=cluster_df, daily_stats=daily_stats, feature_df=feature_df)

if __name__ == "__main__":
    main()

# Запуск из консоли    
#$env:PYTHONUTF8=1; python main.py