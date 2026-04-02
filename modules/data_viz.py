import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def graficar_correlacion(df: pd.DataFrame) -> str | None:
    """
    Genera un heatmap de correlaciones entre columnas numéricas.
    Guarda temp_correlacion.png en el directorio de trabajo actual.
    Devuelve None si todo salió bien, o un mensaje de error.
    """
    try:
        if df is None or df.empty:
            return "No hay datos para correlacionar."
        numericas = df.select_dtypes(include=[np.number])
        if numericas.shape[1] == 0:
            return "No hay columnas numéricas para calcular correlaciones."

        corr = numericas.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
        plt.title("Correlación entre variables numéricas")
        plt.tight_layout()
        ruta = os.path.abspath("temp_correlacion.png")
        plt.savefig(ruta, dpi=120)
        plt.close()
        return None
    except Exception as exc:
        return f"No pude generar el mapa de correlación: {exc}"


def graficar_importancia_caracteristicas(modelo, columnas) -> str | None:
    """
    Gráfico de barras con importancia de características (RandomForest de sklearn).
    Guarda temp_importancia.png.
    """
    try:
        importances = getattr(modelo, "feature_importances_", None)
        if importances is None:
            return "El modelo no expone importancia de características (feature_importances_)."
        columnas = list(columnas)
        if len(columnas) != len(importances):
            return (
                f"Las columnas ({len(columnas)}) no coinciden con el modelo ({len(importances)})."
            )

        orden = np.argsort(importances)[::-1]
        etiquetas = [columnas[i] for i in orden]
        valores = importances[orden]

        plt.figure(figsize=(10, max(4, len(etiquetas) * 0.35)))
        plt.barh(range(len(etiquetas)), valores[::-1], color="steelblue")
        plt.yticks(range(len(etiquetas)), etiquetas[::-1])
        plt.xlabel("Importancia")
        plt.title("Importancia de características (Random Forest)")
        plt.tight_layout()
        ruta = os.path.abspath("temp_importancia.png")
        plt.savefig(ruta, dpi=120)
        plt.close()
        return None
    except Exception as exc:
        return f"No pude generar el gráfico de importancia: {exc}"
