import logging
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Logging setting
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )
    logging.getLogger(__name__).info("Logging is configured.")

logger = logging.getLogger("work_pattern_analyzer")

class Environment(Enum):
    COLAB = 'colab'
    LOCAL = 'local'

@dataclass
class ProjectConfig:
    """
    Оркестратор настроек проекта и менеджер путей файловой системы.
    
    Реализует паттерн "Единый источник истины" (SSOT) для констант окружения. 
    Обеспечивает автоматическое определение среды исполнения (Local vs. Colab) 
    и инкапсулирует кроссплатформенные абстракции путей.
    """
    environment: Environment = Environment.LOCAL
    mount_point: str = '/content/drive/'
    base_path: Path = Path('./data')
    base_path_colab = "/content/drive/MyDrive/Colab Notebooks/data"
    data_file: str = 'timesheet.xlsx'

    def __post_init__(self):
        self._validate_paths()

    def _validate_paths(self):
        if self.environment == Environment.COLAB and not self.mount_point:
            raise ValueError("Mount points обязательно для окружения колаб")
        
    @property
    def data_path(self) -> Path:
        if self.environment == Environment.COLAB:
            return Path(self.base_path_colab) / self.data_file
        return self.base_path / self.data_file
    

RAW_COLUMNS = {'Неделя', 'Дата', 'С', 'По', 'Часы', 'Описание', 'NEW_Тип Активности'}
PROCESSED_COLUMNS = {'date', 'start_time', 'end_time', 'duration_hours', 'activity_type'}
FEATURE_COLUMNS_REQUIRED = PROCESSED_COLUMNS | {'activity_category'}
WEIGHTS = {'deep_work': 2.0, 'training': 1.5, 'meeting': 1.0, 'other': 0.5}
CLUSTER_FEATURES = ['focus_density', 'session_depth']
CATEGORY_MAP = {
    'deep_work': {
        'бизнес-анализ', 'прототипирование', 'методология (документация)',
        'настройка доступов в модели', 'проектный менеджмент', 'работа с воркспейсами',
        'тестирование и доработки', 'моделирование', 'настройка интеграции',
        'подготовка тренинговых материалов'
    },
    'training': {
        'изучение материалов, тренингов, лайфхаков',
    },
    'meeting': {
        'тематические и индивидуальные созвоны', 'организация внутренних мероприятий',
        'созвоны internal projects', 'дейли', 'созвоны bu (business units)',
        'общие созвоны',
    }
}