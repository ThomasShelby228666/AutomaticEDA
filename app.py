import gradio as gr
import pandas as pd
import tempfile
import os

from src.analyzer import DataAnalyzer
from src.visualizer import DataVisualizer

class EDAApp:
    def __init__(self):
        self.visualizer = DataVisualizer()

    def analyze(self, file_obj):
        if file_obj is None:
            raise gr.Error("Пожалуйста, загрузите CSV файл")

        try:
            df = pd.read_csv(file_obj.name)
        except Exception as e:
            raise gr.Error(f"Ошибка чтения файла: {e}")

        analyzer = DataAnalyzer()

        stats_df, info_df = analyzer.get_overview()

        encoded_df = analyzer.preprocess_for_correlation()
        corr_matrix = analyzer.get_correlation_matrix()

        fig_missing = self.visualizer.plot_missing_values(df)
        fig_corr = self.visualizer.plot_correlation_heatmap(corr_matrix)
        dist_figs = self.visualizer.plot_distributions(df)

        paths = []
        with tempfile.TemporaryDirectory() as tmpdir:
            p_corr = os.path.join(tmpdir, "correlation.png")
            fig_corr.savefig(p_corr, bbox_inches="tight")
            paths.append(p_corr)

            p_miss = os.path.join(tmpdir, "missing.png")
            fig_missing.savefig(p_miss, bbox_inches="tight")
            paths.append(p_miss)

            for i, fig in enumerate(dist_figs):
                p_dist = os.path.join(tmpdir, f"dist_{i}.png")
                fig.savefig(p_dist, bbox_inches='tight')
                paths.append(p_dist)

            output_dir = tempfile.mkdtemp()
            final_paths = []

            for p in paths:
                new_name = os.path.join(output_dir, os.path.basename(p))
                os.rename(p, new_name)
                final_paths.append(new_name)

        return (
            df.head(10),
            info_df,
            stats_df,
            fig_missing,
            fig_corr,
            final_paths
        )