import pandas as pd
from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import webbrowser
from threading import Timer

from dashboard import calculate_annual_summary

def run_dashboard(clust_df, daily_stats, feature_df):
    """
    Основная функция запуска визуализации. 
    Принимает данные из пайплайна в main.py
    """
    app = Dash(__name__)
    
    annual_summary = calculate_annual_summary(daily_stats, feature_df)

    # --- Подготовка  ---
    ru_days = {"Monday": "Пн", "Tuesday": "Вт", "Wednesday": "Ср", "Thursday": "Чт", "Friday": "Пт", "Saturday": "Сб", "Sunday": "Вс"}
    ru_day_order = ["Пн", "Вт", "Ср", "Чт", "Пт"]
    cluster_order = ['Пик продуктивности', 'Высокая интенсивность', 'Сбалансированный режим', 'Поддерживающие задачи', 'Операционка и паузы']

    style_dow = (pd.crosstab(clust_df['cluster_label'],clust_df['day_of_week'],normalize='index')
                .rename(columns=ru_days)
                .reindex(index=cluster_order, columns=ru_day_order)
                .mul(100)
                .round(1)
    )
    style_dow_long = style_dow.reset_index().melt(
        id_vars='cluster_label',
        var_name='day_of_week',
        value_name='percent'
    )
    style_dow = style_dow.loc[style_dow.sum(axis=1) > 0]
    style_dow_long['percent_display'] = style_dow_long['percent'].round(1)
    style_dow_long['day_of_week_ru'] = style_dow_long['day_of_week'].map(ru_days)
    heatmap_fig = px.imshow(style_dow, text_auto=".1f", aspect="auto", 
                            color_continuous_scale=[
                                [0.0, "#ecf0f1"],
                                [0.4, "#bdc3c7"],
                                [0.7, "#7f8c8d"],
                                [1.0, "#2c3e50"]
                            ]
    )
    heatmap_fig.update_layout(
        title=dict(text="Распределение стилей работы по дням недели", x=0.5, font=dict(size=16, weight="bold")),
        xaxis_title="День недели",
        yaxis_title="Стиль работы",
        coloraxis_colorbar=dict(title="Доля дней, %", ticksuffix="%"),
        margin=dict(l=40, r=40, t=60, b=40),
        height=360,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    # --- UI Helper Functions ---
    def format_time(val):
        if val is None or pd.isna(val): return "00:00"
        h = int(val)
        m = int((val - h) * 60)
        return f"{h:02d}:{m:02d}"

    def get_color(current_val, threshold, inverse=False):
        pos_color = "#27ae60"
        neg_color = "#e74c3c"
        default_color = "#2c3e50"

        if inverse:
            return neg_color if current_val > threshold else pos_color
        else:
            return pos_color if current_val > threshold else neg_color

    def create_kpi_card(title, value, description=None, color="#2c3e50"):
        return html.Div([
            html.Div([
                html.H3(value, style={'margin': '0', 'color': color, 'fontSize': '1.8rem', 'fontWeight': 'bold'}),
                html.P(title, style={'margin': '4px 0', 'color': '#34495e', 'fontSize': '0.9rem', 'fontWeight': '600', 'textTransform': 'uppercase'}),
                html.Div([
                    html.Small(description, style={'color': '#95a5a6', 'fontSize': '0.75rem', 'lineHeight': '1.2'})
                ], style={'marginTop': '8px', 'minHeight': '20px', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'})
            ], style={
                'background': 'white', 'padding': '12px', 'borderRadius': '12px',
                'boxShadow': '0 4px 6px rgba(0,0,0,0.05)', 'textAlign': 'center',
                'borderTop': f'4px solid {color}', 'height': '100%'
            })
        ], style={'padding': '5px'})

    # --- Layout ---
    app.layout = html.Div([
        # Шапка
        html.Div([
            html.H1("PRODUCTIVITY PERFORMANCE REVIEW 2025",
                    style={'margin': '0', 'color': '#2c3e50', 'letterSpacing': '2px', 'fontWeight': '800'}),
            html.P(f"Комплексный анализ структуры времени и когнитивной эффективности",
                style={'margin': '5px 0 25px 0', 'color': '#7f8c8d', 'fontSize': '1rem'})
        ], style={'textAlign': 'center', 'fontFamily': 'Segoe UI, Roboto, Helvetica, Arial, sans-serif'}),

        # Слой 1: KPI Карточки
        html.Div([
            create_kpi_card("Часов", f"{float(annual_summary['main_metrics']['total_hours']['value']):.1f}", annual_summary['main_metrics']['total_hours']['desc'], "#2c3e50"),
            create_kpi_card("Среднее за день", f"{float(annual_summary['main_metrics']['avg_daily_hours']['value']):.1f}ч", annual_summary['main_metrics']['avg_daily_hours']['desc'], "#2c3e50"),
            create_kpi_card(
                "Стабильность",
                f"{float(annual_summary['spetial_metrics']['consistency_score']['value']):.2f}ч",
                annual_summary['spetial_metrics']['consistency_score']['desc'],
                get_color(annual_summary['spetial_metrics']['consistency_score']['value'], 1.2, inverse=True)
            ),
            create_kpi_card(
                "Обучения",
                f"{float(annual_summary['spetial_metrics']['learning_index']['value']):.1f}%",
                annual_summary['spetial_metrics']['learning_index']['desc'],
                get_color(annual_summary['spetial_metrics']['learning_index']['value'], 5)
            ),
        ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(4, 1fr)', 'gap': '10px', 'marginBottom': '20px'}),
        html.Div([
            create_kpi_card(
                "Глубоких сессий",
                f"{float(annual_summary['spetial_metrics']['makers_index']['value']):.1f}%",
                annual_summary['spetial_metrics']['makers_index']['desc'],
                get_color(annual_summary['spetial_metrics']['makers_index']['value'], 40)
            ),
            create_kpi_card(
                "Коэффициент Deep Work",
                f"{float(annual_summary['spetial_metrics']['deep_work_ratio']['value']):.2f}",
                annual_summary['spetial_metrics']['deep_work_ratio']['desc'],
                get_color(annual_summary['spetial_metrics']['deep_work_ratio']['value'], 1.5)
            ),
            create_kpi_card(
                "Фрагментация",
                f"{float(annual_summary['spetial_metrics']['fragmentation_index']['value']):.2f}",
                annual_summary['spetial_metrics']['fragmentation_index']['desc'],
                get_color(annual_summary['spetial_metrics']['fragmentation_index']['value'], 2.5, inverse=True)
            ),
            create_kpi_card("Типов активности", f"{float(annual_summary['spetial_metrics']['unique_activity']['value']):.1f}", annual_summary['spetial_metrics']['unique_activity']['desc'], "#2c3e50"),
        ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(4, 1fr)', 'gap': '10px', 'marginBottom': '25px'}),

    # Слой 2: Детализация категорий
    html.Div([
        html.Div([
            html.H4("ДЕТАЛИЗАЦИЯ ПО КАТЕГОРИЯМ",
                    style={
                        'height': '60px',
                        'display': 'flex',
                        'alignItems': 'center',
                        'paddingLeft': '25px',
                        'margin': '0',
                        'color': '#34495e',
                        'fontSize': '1.1rem',
                        'letterSpacing': '1px',
                        'fontWeight': '700'
                    }),
            dcc.Tabs(id="category-tabs", value='deep_work', children=[
                dcc.Tab(label='DEEP WORK', value='deep_work', style={'fontWeight': 'bold'}, selected_style={'fontWeight': 'bold', 'borderTop': '3px solid #2c3e50'}),
                dcc.Tab(label='MEETINGS', value='meeting', style={'fontWeight': 'bold'}, selected_style={'fontWeight': 'bold', 'borderTop': '3px solid #2c3e50'}),
                dcc.Tab(label='TRAINING', value='training', style={'fontWeight': 'bold'}, selected_style={'fontWeight': 'bold', 'borderTop': '3px solid #2c3e50'}),
                dcc.Tab(label='OTHER', value='other', style={'fontWeight': 'bold'}, selected_style={'fontWeight': 'bold', 'borderTop': '3px solid #2c3e50'}),
            ], style={'height': '50px'}),
            html.Div(id='tabs-content', style={'padding': '20px'})
        ], style={
            'background': 'white',
            'borderRadius': '12px',
            'boxShadow': '0 4px 15px rgba(0,0,0,0.05)',
            'overflow': 'hidden'
        })
    ], style={'marginTop': '35px'}),

    html.Div([
        html.Div([
            html.H4(
                "ХАРАКТЕРИСТИКА СТИЛЕЙ РАБОТЫ ПО ДНЯМ НЕДЕЛИ",
                style={
                        'height': '60px',
                        'display': 'flex',
                        'alignItems': 'center',
                        'paddingLeft': '25px',
                        'margin': '0',
                        'color': '#34495e',
                        'fontSize': '1.1rem',
                        'letterSpacing': '1px',
                        'fontWeight': '700'
                }
            ),
            dcc.Graph(
                figure=heatmap_fig,
                config={'displayModeBar': False}
            )
        ], style={
            'background': 'white',
            'padding': '20px',
            'borderRadius': '12px',
            'boxShadow': '0 4px 15px rgba(0,0,0,0.05)',
            'marginBottom': '30px'
        })
    ], style={'marginTop': '10px'})

    ], style={'padding': '40px', 'backgroundColor': '#f0f2f5', 'minHeight': '100vh', 'fontFamily': 'sans-serif', 'maxWidth': '1200px', 'margin': '0 auto'})


    # === Категории активнсотей
    @app.callback(
        Output('tabs-content', 'children'),
        Input('category-tabs', 'value')
    )
    def render_content(tab_key):
        try:
            cat_data = annual_summary['categories'].get(tab_key, {})
            filtered = feature_df[feature_df['activity_category'] == tab_key].copy()

            if filtered.empty:
                return html.Div("Нет данных для отображения", style={'textAlign': 'center', 'padding': '40px', 'color': '#bdc3c7'})

            plot_df = filtered.groupby('activity_type')['duration_hours'].sum().reset_index()
            plot_df = plot_df.sort_values(by='duration_hours', ascending=True)

            fig = px.bar(
                plot_df, x='duration_hours', y='activity_type', orientation='h',
                text_auto='.1f', color_discrete_sequence=['#2c3e50']
            )

            fig.update_traces(
                hovertemplate="Тип активности: %{y}<br>" +
                            "Часы: %{x:.1f}<extra></extra>"
            )

            fig.update_layout(
                margin=dict(l=10, r=40, t=10, b=10), height=300,
                xaxis_title="Суммарные часы", yaxis_title=None,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                xaxis={'showgrid': True, 'gridcolor': '#f0f0f0'}
            )

            return html.Div([
                # Мини-статистика вкладки
                html.Div([
                    html.Div([
                        html.P("Всего часов", style={'color': '#7f8c8d', 'margin': '0', 'fontSize': '0.8rem'}),
                        html.H4(f"{float(cat_data.get('hours', 0)):.1f}", style={'margin': '0', 'color': '#2c3e50'})
                    ], style={'flex': '1', 'textAlign': 'center', 'borderRight': '1px solid #f0f0f0'}),
                    html.Div([
                        html.P("Кол-во сессий", style={'color': '#7f8c8d', 'margin': '0', 'fontSize': '0.8rem'}),
                        html.H4(f"{int(cat_data.get('sessions', 0))}", style={'margin': '0', 'color': '#2c3e50'})
                    ], style={'flex': '1', 'textAlign': 'center', 'borderRight': '1px solid #f0f0f0'}),
                    html.Div([
                        html.P("Доля от общего", style={'color': '#7f8c8d', 'margin': '0', 'fontSize': '0.8rem'}),
                        html.H4(f"{float(cat_data.get('persentage', 0)):.1f}%", style={'margin': '0', 'color': '#2c3e50'})
                    ], style={'flex': '1', 'textAlign': 'center'}),
                ], style={'display': 'flex', 'background': '#f8f9fa', 'padding': '15px', 'borderRadius': '8px', 'marginBottom': '20px'}),

                dcc.Graph(figure=fig, config={'displayModeBar': False})
            ])
        except Exception as e:
            return html.Div(f"Ошибка: {e}")

    Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:8050")).start()
    
    print(">>> Дашборд запущен на http://127.0.0.1:8050")
    app.run(debug=False, port=8050, use_reloader=False)