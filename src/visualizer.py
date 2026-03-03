import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List

class DataVisualizer:
    """
    Отвечает за построение графиков.
    Он не зависит от Gradio, возвращает объекты Figure.
    """
    def __init__(self, style: str = "whitegrid"):
        sns.set_theme(style=style)
        plt.rcParams["figure.figsize"] = (10, 6)

    def plot_missing_values(self, df: pd.DataFrame) -> plt.Figure:
        """
        Строит тепловую карту пропусков.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="viridis", ax=ax)
        ax.set_title("Карта пропущенных значений", fontsize=15)
        return fig

    def plot_correlation_heatmap(self, corr_matrix: pd.DataFrame) -> plt.Figure:
        """
        Строит корреляционную матрицу.
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                    linewidths=.5, ax=ax, center=0)
        ax.set_title("Корреляционная матрица (включая кодированные категории)", fontsize=15)
        return fig

    def plot_disrobutions(self, df: pd.DataFrame, max_cols: int = 9) -> List[plt.Figure]:
        """
        Строит гистограммы для числовых признаков.
        Возвращает список фигур, так как графиков может быть много.
        """
        figures = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:max_cols]

        for col in numeric_cols:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df[col], kde=True, ax=ax, color="skyblue", bins=30)
            ax.set_title(f"Распределение: {col}", fontsize=12)
            ax.set_xlabel(col)
            ax.set_ylabel("Частота")
            figures.append(fig)

        return figures