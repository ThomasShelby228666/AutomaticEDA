import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple

class DataAnalyzer:
    """
    Класс отвечает за вычислительную часть EDA:
    - Статистика
    - Поиск пропусков
    - Кодирование категорий для корреляции
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_df = None

    def get_overview(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Возвращает общую статистику и информацию о типах.
        """
        stats = self.df.describe(include="all").transpose()

        info_df = pd.DataFrame({
            "Тип данных": self.df.dtypes,
            "Пропуски": self.df.isnull().sum(),
            "% пропусков": self.df.isnull().sum() / self.df.shape[0] * 100,
            "Уникальные": self.df.nunique()
        })

        return stats, info_df

    def preprocess_for_correlation(self) -> pd.DataFrame:
        """
        Подготавливает данные для корреляционной матрицы.
        Кодирует категориальные признаки с помощью LabelEncoder.
        """
        df_encoded = self.df.copy()

        for col in df_encoded.select_dtypes(include=["object", "category"]).columns:
            df_encoded[col] = df_encoded[col].fillna("MISSING")
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

        for col in df_encoded.select_dtypes(include=[np.number]).columns:
            df_encoded[col] = df_encoded[col].fillna(df_encoded[col].median())

        self.numeric_df = df_encoded

        return df_encoded

    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        Считает корреляцию Пирсона.
        """
        if self.numeric_df is None:
            self.preprocess_for_correlation()

        return self.numeric_df.corr()
