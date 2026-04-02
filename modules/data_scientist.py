import os

import pandas as pd

# Umbral heurístico: por debajo se considera variable discreta / pocos valores únicos.
_MAX_UNICOS_CLASIFICACION = 20


def objetivo_es_clasificacion(df: pd.DataFrame) -> bool:
    """
    True si la última columna (objetivo) encaja con un problema de clasificación;
    False si encaja mejor con regresión. Misma heurística que sugerir_modelo.
    """
    if df is None or df.empty or len(df.columns) == 0:
        return True

    serie = df[df.columns[-1]]
    serie_limpia = serie.dropna()
    n_filas = len(serie_limpia)
    n_unicos = serie_limpia.nunique() if n_filas else 0

    es_objeto_o_texto = (
        pd.api.types.is_object_dtype(serie)
        or pd.api.types.is_string_dtype(serie)
        or pd.api.types.is_categorical_dtype(serie)
    )

    pocos_valores_unicos = n_filas > 0 and n_unicos <= _MAX_UNICOS_CLASIFICACION

    if es_objeto_o_texto or pocos_valores_unicos:
        return True
    if pd.api.types.is_numeric_dtype(serie):
        return False
    return True


def sugerir_modelo(df: pd.DataFrame) -> str:
    """
    Infiere si conviene Clasificación o Regresión según la última columna (objetivo)
    y sus dtypes / cardinalidad.
    """
    if df is None or df.empty or len(df.columns) == 0:
        return "No puedo sugerir un tipo de modelo: el DataFrame no tiene columnas útiles."

    objetivo = df.columns[-1]
    serie = df[objetivo]
    dtype = serie.dtype

    nulos = int(serie.isnull().sum())
    serie_limpia = serie.dropna()
    n_filas = len(serie_limpia)
    n_unicos = serie_limpia.nunique() if n_filas else 0

    es_objeto_o_texto = (
        pd.api.types.is_object_dtype(serie)
        or pd.api.types.is_string_dtype(serie)
        or pd.api.types.is_categorical_dtype(serie)
    )

    pocos_valores_unicos = n_filas > 0 and n_unicos <= _MAX_UNICOS_CLASIFICACION

    if es_objeto_o_texto or pocos_valores_unicos:
        tipo = "Clasificación"
        razon = (
            "la columna objetivo es categórica/texto o tiene pocos valores distintos "
            "(variable discreta / etiquetas)."
        )
    elif pd.api.types.is_numeric_dtype(serie):
        tipo = "Regresión"
        razon = (
            "la columna objetivo es numérica con cardinalidad alta respecto al tamaño "
            "(comportamiento típico de un objetivo continuo o casi continuo)."
        )
    else:
        tipo = "Clasificación"
        razon = "el tipo de dato del objetivo encaja mejor con etiquetas discretas."

    return (
        f"Recomendación técnica: plantear un problema de {tipo}. "
        f"Criterio: {razon} "
        f"(columna objetivo «{objetivo}», dtype {dtype}, valores únicos ≈ {n_unicos}"
        + (f", celdas nulas en objetivo: {nulos}" if nulos else "")
        + ")."
    )


def analizar_dataset(ruta_archivo: str) -> tuple[str, pd.DataFrame | None]:
    """
    Lee un archivo tabular (CSV o Excel) con pandas y devuelve
    (resumen + recomendación, DataFrame o None si falla).
    """
    ruta = os.path.expanduser((ruta_archivo or "").strip().strip('"').strip("'"))
    if not ruta:
        return "No se indicó una ruta de archivo válida.", None

    _, ext = os.path.splitext(ruta)
    ext = ext.lower()

    if ext not in (".csv", ".xls", ".xlsx"):
        return "Formato no soportado. Por favor usa CSV o Excel", None

    try:
        if ext == ".csv":
            df = pd.read_csv(ruta)
        else:
            df = pd.read_excel(ruta)
    except FileNotFoundError:
        return f"No encontré el archivo: {ruta}", None
    except pd.errors.EmptyDataError:
        return f"El archivo está vacío o no tiene datos tabulares: {ruta}", None
    except UnicodeDecodeError:
        return (
            f"No pude decodificar el archivo como texto (encoding). "
            f"Prueba a guardar el CSV como UTF-8: {ruta}"
        ), None
    except Exception as exc:
        return f"No pude leer el archivo. Detalle: {exc}", None

    n_filas = len(df)
    columnas = list(df.columns)
    nulos_total = int(df.isnull().sum().sum())
    hay_nulos = nulos_total > 0

    nombres = ", ".join(str(c) for c in columnas)
    parte_nulos = (
        "No hay valores nulos."
        if not hay_nulos
        else f"Sí hay valores nulos (celdas nulas en total: {nulos_total})."
    )

    resumen = (
        f"El dataset tiene {n_filas} filas, columnas: {nombres}. {parte_nulos}"
    )
    recomendacion = sugerir_modelo(df)
    texto = f"{resumen}\n\n{recomendacion}"
    return texto, df
