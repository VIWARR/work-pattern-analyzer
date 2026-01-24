def calculate_annual_summary(daily_stats, feature_df):
    annual_summary = {
        'main_metrics': {
            'total_hours': {
                'value': daily_stats['total_hours'].sum(),
                'desc': "Общий объем отработанных часов за год"
            },
            'avg_daily_hours': {
                'value': daily_stats['total_hours'].mean(),
                'desc': "Средняя продолжительность рабочего дня в часах"
            },
            'count_sessions': daily_stats['sessions_count'].sum(),
        },
        'spetial_metrics': {
            'flow_session': {
                'value': daily_stats['is_flow_session'].sum(),
                'desc': "Сессии Deep Work длительностью более 1 часа"
            },
            'fragmentation_index': {
                'value': daily_stats['sessions_count'].sum() / daily_stats['total_hours'].sum(),
                'desc': "Среднее кол-во переключений между активностями за 1 час работы"
            },
            'late_sessions': {
                'value': daily_stats['is_overtime'].sum(),
                'desc': "Количество задач, начатых после 19.00"
            },
            'deep_work_ratio': {
                'value': daily_stats['hours_deep_work'].sum() / max((daily_stats['hours_meeting'].sum() + daily_stats['hours_other'].sum()), 1),
                'desc':"Количество часов Deep Work, приходящихся на 1 час категорий meeting и other"
            },
            'makers_index': {
                'value': daily_stats['makers_hours'].sum() / daily_stats['total_hours'].sum() * 100,
                'desc': "Процент времени проведенного в сессиях Deep Work более 1 часа"
            },
            'consistency_score': {
                'value': daily_stats['start_hour'].std(),
                'desc': "Стандартное отклонение времени начала рабочего дня"
            },
            'learning_index': {
                'value': daily_stats['hours_training'].sum() / daily_stats['total_hours'].sum() * 100,
                'desc': "Процент времени затраченного на обучение"
            },
            'unique_activity': {
                'value': daily_stats['unique_activities'].mean(),
                'desc': "Среднее количество разных типов активности в течение одного дня"
            }
        },
        'categories': {
            'deep_work': {
                'hours': daily_stats['hours_deep_work'].sum(),
                'sessions': daily_stats['sessions_deep_work'].sum(),
                'persentage': daily_stats['hours_deep_work'].sum() / daily_stats['total_hours'].sum() * 100
            },
            'meeting': {
                'hours': daily_stats['hours_meeting'].sum(),
                'sessions': daily_stats['sessions_meeting'].sum(),
                'persentage': daily_stats['hours_meeting'].sum() / daily_stats['total_hours'].sum() * 100
            },
            'training': {
                'hours': daily_stats['hours_training'].sum(),
                'sessions': daily_stats['sessions_training'].sum(),
                'persentage': daily_stats['hours_training'].sum() / daily_stats['total_hours'].sum() * 100
            },
            'other': {
                'hours': daily_stats['hours_other'].sum(),
                'sessions': daily_stats['sessions_other'].sum(),
                'persentage': daily_stats['hours_other'].sum() / daily_stats['total_hours'].sum() * 100
            },
        },
    }
    return annual_summary