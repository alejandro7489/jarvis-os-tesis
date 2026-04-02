import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder

from modules.data_scientist import objetivo_es_clasificacion

# Búsqueda de hiperparámetros (RandomForest): equilibrio entre calidad y tiempo de cómputo.
_PARAM_GRID_RF = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20, 30],
}


def _cv_splits(n_train: int) -> int:
    """Pliegues de CV acotados al tamaño del conjunto de entrenamiento."""
    if n_train < 6:
        return 2
    return min(5, max(3, n_train // 3))


def _rellenar_nulos(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].fillna(out[col].mean())
        else:
            modo = out[col].mode()
            relleno = modo.iloc[0] if len(modo) else ""
            out[col] = out[col].fillna(relleno)
    return out


def _encode_texto_a_numeros(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    out = df.copy()
    encoders: dict[str, LabelEncoder] = {}
    for col in out.columns:
        if (
            pd.api.types.is_object_dtype(out[col])
            or pd.api.types.is_string_dtype(out[col])
            or pd.api.types.is_categorical_dtype(out[col])
        ):
            le = LabelEncoder()
            out[col] = le.fit_transform(out[col].astype(str))
            encoders[col] = le
    return out, encoders


def entrenar_modelo_rapido(df: pd.DataFrame) -> tuple[str, object | None, list | None]:
    """
    Entrena un Random Forest (clasificación o regresión) con GridSearchCV
    sobre n_estimators y max_depth.
    Devuelve (mensaje, modelo_entrenado_o_None, nombres_columnas_X_o_None).
    Los LabelEncoder de features quedan en modelo._jarvis_label_encoders (dict).
    Si la última columna (y) era categórica y se codificó, el encoder queda en
    modelo._jarvis_target_encoder.
    """
    try:
        if df is None or df.empty:
            return "No hay datos para entrenar (DataFrame vacío).", None, None
        if len(df.columns) < 2:
            return (
                "Se necesitan al menos dos columnas: características y columna objetivo (la última).",
                None,
                None,
            )

        datos = _rellenar_nulos(df)
        datos, encoders = _encode_texto_a_numeros(datos)

        y = datos.iloc[:, -1]
        X = datos.iloc[:, :-1]

        if len(X.columns) == 0:
            return (
                "No hay columnas de características (solo existe la columna objetivo).",
                None,
                None,
            )

        nombres_columnas = list(X.columns)
        encoders_x = {c: encoders[c] for c in nombres_columnas if c in encoders}
        nombre_objetivo = datos.columns[-1]
        encoder_y = encoders.get(nombre_objetivo)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        es_clasificacion = objetivo_es_clasificacion(df)
        cv = _cv_splits(len(X_train))

        if es_clasificacion:
            grid = GridSearchCV(
                RandomForestClassifier(random_state=42, n_jobs=1),
                _PARAM_GRID_RF,
                cv=cv,
                scoring="accuracy",
                n_jobs=-1,
                refit=True,
            )
            grid.fit(X_train, y_train)
            modelo = grid.best_estimator_
            pred = modelo.predict(X_test)
            metrica = accuracy_score(y_test, pred)
            pct = metrica * 100
            msg = (
                f"Entrenamiento finalizado. El modelo alcanzó un {pct:.0f}% de precisión."
            )
            modelo._jarvis_label_encoders = encoders_x
            modelo._jarvis_target_encoder = encoder_y
            return msg, modelo, nombres_columnas

        grid = GridSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=1),
            _PARAM_GRID_RF,
            cv=cv,
            scoring="r2",
            n_jobs=-1,
            refit=True,
        )
        grid.fit(X_train, y_train)
        modelo = grid.best_estimator_
        pred = modelo.predict(X_test)
        metrica = r2_score(y_test, pred)
        pct = metrica * 100
        msg = (
            f"Entrenamiento finalizado. El modelo alcanzó un {pct:.0f}% de precisión predictiva."
        )
        modelo._jarvis_label_encoders = encoders_x
        modelo._jarvis_target_encoder = encoder_y
        return msg, modelo, nombres_columnas
    except Exception as exc:
        return f"No pude completar el entrenamiento. Detalle: {exc}", None, None


def hacer_prediccion_unica(valores_texto: str, modelo, columnas_referencia: list):
    """
    Parsea una línea CSV de valores, aplica los LabelEncoder de X
    (modelo._jarvis_label_encoders) y devuelve (predicción, error).
    Si existía encoder de y (modelo._jarvis_target_encoder), devuelve la etiqueta original en texto.
    """
    try:
        encoders = getattr(modelo, "_jarvis_label_encoders", None) or {}
        columnas_referencia = list(columnas_referencia)
        if not valores_texto or not str(valores_texto).strip():
            return None, "No recibí valores para predecir."
        if modelo is None:
            return None, "No hay modelo cargado."
        if not columnas_referencia:
            return None, "No hay columnas de referencia para las características."

        linea = str(valores_texto).strip().strip('"').strip("'")
        partes = [p.strip() for p in linea.split(",")]
        if len(partes) != len(columnas_referencia):
            return None, (
                f"Se esperaban {len(columnas_referencia)} valores (uno por columna), "
                f"se recibieron {len(partes)}. Orden: {', '.join(columnas_referencia)}."
            )

        fila: list[float] = []
        for col, raw in zip(columnas_referencia, partes, strict=True):
            if col in encoders:
                le = encoders[col]
                try:
                    codificado = float(le.transform([str(raw)])[0])
                except ValueError as exc:
                    return None, (
                        f"No reconozco el valor «{raw}» para «{col}» dentro de las clases del entrenamiento. "
                        f"Detalle: {exc}"
                    )
                fila.append(codificado)
            else:
                try:
                    texto = str(raw).strip().replace(",", ".")
                    fila.append(float(texto))
                except (TypeError, ValueError):
                    return None, f"No pude interpretar «{raw}» como número para la columna «{col}»."

        X = np.asarray([fila], dtype=np.float64)
        pred = modelo.predict(X)
        if pred is None or len(pred) == 0:
            return None, "El modelo no devolvió una predicción."

        valor_crudo = pred[0]
        target_enc = getattr(modelo, "_jarvis_target_encoder", None)
        if target_enc is not None:
            try:
                idx = int(np.rint(float(valor_crudo)))
                etiqueta = target_enc.inverse_transform(np.array([idx]))[0]
            except (ValueError, TypeError) as exc:
                return None, f"No pude convertir la predicción a etiqueta original: {exc}"
            return str(etiqueta), None

        return valor_crudo, None
    except Exception as exc:
        return None, f"Error al predecir: {exc}"
