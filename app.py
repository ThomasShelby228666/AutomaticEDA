import gradio as gr
import pandas as pd
import tempfile
import os
import matplotlib.pyplot as plt

from src.analyzer import DataAnalyzer
from src.visualizer import DataVisualizer

class EDAApp:
    """
    Главный класс, отвечает за сам EDA.
    Объединяет анализ данных и визуализацию.
    """
    def __init__(self):
        self.visualizer = DataVisualizer()

    def analyze(self, file_obj):
        """
        Полный анализ загруженного CSV-файла.
        Метод возвращает результаты, подходящие для визуализации в Gradio.
        """
        if file_obj is None:
            raise gr.Error("Пожалуйста, загрузите CSV файл")

        try:
            df = pd.read_csv(file_obj.name)
        except Exception as e:
            raise gr.Error(f"Ошибка чтения файла: {e}")

        analyzer = DataAnalyzer(df)

        # 1. Общая статистика
        stats_df, info_df = analyzer.get_overview()

        # 2. Препроцессинг и корреляции
        encoded_df = analyzer.preprocess_for_correlation()
        corr_matrix = analyzer.get_correlation_matrix()

        # 3. Построение графиков
        fig_missing = self.visualizer.plot_missing_values(df)
        fig_corr = self.visualizer.plot_correlation_heatmap(corr_matrix)
        dist_figs = self.visualizer.plot_distributions(df)

        output_dir = tempfile.mkdtemp()
        dist_paths = []

        for i, fig in enumerate(dist_figs):
            p_dist = os.path.join(output_dir, f"dist_{i}.png")
            fig.savefig(p_dist, bbox_inches="tight")
            dist_paths.append(p_dist)

        for fig in dist_figs:
            plt.close(fig)

        return (
            df.head(10),  # Превью
            info_df,  # Инфо
            stats_df,  # Статистика
            fig_missing,  # Пропуски
            fig_corr,  # Корреляция
            dist_paths  # Распределения
        )
        # Сохранение графиков во временные файлы для скачивания
        # path_corr = None
        # path_missing = None
        # dist_paths = []
        # with tempfile.TemporaryDirectory() as tmpdir:
        #     p_corr = os.path.join(tmpdir, "correlation.png")
        #     fig_corr.savefig(p_corr, bbox_inches="tight")
        #     path_corr = p_corr
        #
        #     p_miss = os.path.join(tmpdir, "missing.png")
        #     fig_missing.savefig(p_miss, bbox_inches="tight")
        #     path_missing = p_miss
        #
        #     for i, fig in enumerate(dist_figs):
        #         p_dist = os.path.join(tmpdir, f"dist_{i}.png")
        #         fig.savefig(p_dist, bbox_inches="tight")
        #         dist_paths.append(p_dist)
        #
        #     # Создадим постоянную временную папку для хранения графиков.
        #     output_dir = tempfile.mkdtemp()
        #     final_corr = None
        #     final_missing = None
        #     final_dist_paths = []
        #
        #     if path_corr:
        #         new_name = os.path.join(output_dir, "correlation.png")
        #         os.rename(path_corr, new_name)
        #         final_corr = new_name
        #
        #     if path_missing:
        #         new_name = os.path.join(output_dir, "missing.png")
        #         os.rename(path_missing, new_name)
        #         final_missing = new_name
        #
        #     for p in dist_paths:
        #         new_name = os.path.join(output_dir, os.path.basename(p))
        #         os.rename(p, new_name)
        #         final_dist_paths.append(new_name)
        #
        # return (
        #     df.head(10), # Превью
        #     info_df, # Инфо
        #     stats_df, #  Статистика
        #     fig_missing, # Пропуски
        #     fig_corr, # Корреляция
        #     final_dist_paths # Распределения
        # )

def launch_interface():
    """
    Создаёт и запускает веб-интерфейс Gradio для приложения AutoEDA.
    Функция инициализирует все компоненты интерфейса.
    """
    app = EDAApp()

    # Создание блока Gradio с кастомной темой
    with gr.Blocks(theme=gr.themes.Soft(), title="Auto EDA report") as demo:
        gr.Markdown("**Автоматический EDA**")
        gr.Markdown("Загрузите свой CSV-файл для детального анализа")

        # Секция загрузки файла
        with gr.Row():
            file_input = gr.File(label="Загрузите файл", file_types=[".csv"])

        # Кнопка запуска анализа
        button = gr.Button("Запуск анализа", variant="primary", size="lg")

        # Вкладки с результатами анализа
        with gr.Tabs():
            with gr.TabItem("Обзор данных"):
                with gr.Row():
                    preview_df = gr.DataFrame(label="Первые 10 строк датасета", interactive=False)
                with gr.Row():
                    info_df = gr.DataFrame(label="Информация о датасете", interactive=False)
                with gr.Row():
                    stats_df = gr.DataFrame(label="Описательная статистика", interactive=False)

            with gr.TabItem("Визуализация"):
                gr.Markdown("### Анализ корреляций и пропусков")
                with gr.Row():
                    plot_missing = gr.Plot(label="Пропуски")
                    plot_corr = gr.Plot(label="Матрица корреляций")

                gr.Markdown("### Распределение признаков")
                gallery = gr.Gallery(label="Гистограммы распределений", columns=3, height="auto", object_fit="scale-down")

            button.click(
                fn=app.analyze,
                inputs=[file_input],
                outputs=[preview_df, info_df, stats_df, plot_missing, plot_corr, gallery]
            )

    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    launch_interface()