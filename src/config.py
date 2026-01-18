import logging
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Logging setting
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

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
    raw_data_file: str = 'timesheet.xlsx'
    processed_file: str = 'daily_stats.csv'

    def __post_init(self):
        self._validate_paths()

    @property
    def row_path(self) -> Path:
        return self.base_path / 'raw' / self.raw_data_file
    
    @property
    def processed_path(self) -> Path:
        return self.base_path / 'processed' / self.processed_file
    

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