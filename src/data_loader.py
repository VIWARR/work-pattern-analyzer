import pandas as pd
from pathlib import Path
from src.config import logger, ProjectConfig, RAW_COLUMNS, Environment


class DataLoader:
    """
    Объект доступа к данным (DAO) для операций ввода-вывода.
    
    Отвечает за низкоуровневое чтение потоков и первичную валидацию схемы данных. 
    Реализует паттерн "Стратегия" для выбора парсера на основе расширения файла.
    """
    _SUPPORTED_FORMATS = {
        '.csv': pd.read_csv,
        '.xlsx': pd.read_excel,
        '.xls': pd.read_excel,
        '.parquet': pd.read_parquet,
        '.feather': pd.read_feather
    }

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config

    def load(self, **kwargs) -> pd.DataFrame:
        self._mount_drive_if_needed()
        path = self.config.data_path

        self._validate_file(path)
        loader = self._SUPPORTED_FORMATS[path.suffix.lower()]

        df = loader(path, **kwargs)
        self._validate_dataframe(df)

        logger.info(f"Файл успешно загружен: {len(df)} строк, {len(df.columns)} столбцов")
        return df
    
    def _mount_drive_if_needed(self) -> None:
        if self.config.environment == Environment.COLAB:
            from google.colab import drive
            drive.mount(self.config.mount_point)
    
    def _validate_file(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(path)
        
        if path.suffix.lower() not in self._SUPPORTED_FORMATS:
            raise ValueError(f"Неподдерживаемый формат файла")
        
    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("Датафрейм пустой")
        
        missing_columns = RAW_COLUMNS - set(df.columns)
        if missing_columns:
            raise ValueError(f"Отсутствуют необходимые признаки: {missing_columns}")
        
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Найдено {duplicates} дублирующих строк")

        missing_value = df.isnull().sum().sum()
        if missing_value > 0:
            logger.warning(f"Найдено {missing_value} пропущенных значений.")