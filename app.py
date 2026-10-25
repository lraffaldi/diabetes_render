"""Demo académica UGR: Gradio + scikit-learn, preparada para Render."""

import os
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import gradio as gr
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


VARIABLES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
MEDICIONES = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
NOMBRES = [
    "Embarazos", "Glucosa", "Presión diastólica", "Pliegue del tríceps",
    "Insulina", "IMC", "Función de pedigrí", "Edad"
]
ETIQUETAS = [
    "Cantidad de embarazos", "Glucosa a las 2 horas (mg/dL)",
    "Presión diastólica (mmHg)", "Pliegue del tríceps (mm)",
    "Insulina a las 2 horas (µU/mL)", "IMC (kg/m²)",
    "Función de pedigrí de diabetes (valor del dataset)", "Edad (años)"
]
COLUMNAS_SALIDA = ["Variable", "Aporte al puntaje", "Dato imputado"]


def limpiar_ceros(df):
    """Ceros de mediciones a NaN; cero embarazos se conserva."""
    df = df.copy()
    df[MEDICIONES] = df[MEDICIONES].replace(0, np.nan)
    return df


# El CSV forma parte del proyecto: no requiere Drive ni descargas al iniciar.
datos = pd.read_csv(Path(__file__).with_name("diabetes.csv"))
X = datos[VARIABLES].copy()
y = datos["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Se conserva el entrenamiento del Colab sobre 614 filas, no sobre todo el CSV.
modelo_logistico = make_pipeline(
    FunctionTransformer(limpiar_ceros, validate=False),
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)
modelo_logistico.fit(X_train, y_train)

predicciones = modelo_logistico.predict(X_test)
exactitud = accuracy_score(y_test, predicciones)
sensibilidad = recall_score(y_test, predicciones)
vn, fp, fn, vp = confusion_matrix(y_test, predicciones, labels=[0, 1]).ravel()


def predecir_diabetes(embarazos, glucosa, presion, pliegue_piel,
                     insulina, imc, pedigree, edad):
    """Devuelve resultado y aportes individuales al puntaje logístico."""
    vacia = pd.DataFrame(columns=COLUMNAS_SALIDA)
    try:
        paciente = pd.DataFrame([[
            embarazos, glucosa, presion, pliegue_piel,
            insulina, imc, pedigree, edad
        ]], columns=VARIABLES, dtype=float)
    except (TypeError, ValueError):
        return "Ingresá valores numéricos o dejá vacías las mediciones ausentes.", vacia

    obligatorias = ["Pregnancies", "DiabetesPedigreeFunction", "Age"]
    if paciente[obligatorias].isna().any().any():
        return "Completá embarazos, función de pedigrí y edad.", vacia
    if np.isinf(paciente.to_numpy()).any() or (paciente < 0).any().any():
        return "Ingresá números finitos y no negativos.", vacia
    if (paciente.loc[0, ["Pregnancies", "Age"]] % 1 != 0).any():
        return "Embarazos y edad deben ser números enteros.", vacia
    if paciente.loc[0, "Age"] < 21:
        return "La demo contempla edades de 21 años o más, como el dataset.", vacia

    faltantes = limpiar_ceros(paciente).isna().iloc[0]
    if faltantes[MEDICIONES].all():
        return "Ingresá al menos una medición clínica para esta demo.", vacia

    # La predicción recibe datos crudos: el pipeline aplica todo el tratamiento.
    probabilidad = float(modelo_logistico.predict_proba(paciente)[0, 1])
    clase = int(modelo_logistico.predict(paciente)[0])

    # Aporte = valor estandarizado por coeficiente; no son porcentajes.
    z = modelo_logistico[:-1].transform(paciente)[0]
    coef = modelo_logistico.named_steps["logisticregression"].coef_[0]
    aportes = z * coef
    orden = np.argsort(-np.abs(aportes))
    tabla = pd.DataFrame({
        "Variable": [NOMBRES[i] for i in orden],
        "Aporte al puntaje": aportes[orden].round(3),
        "Dato imputado": ["Sí" if faltantes.iloc[i] else "No" for i in orden]
    })

    campos = [NOMBRES[i] for i in range(len(VARIABLES)) if faltantes.iloc[i]]
    porcentaje = f"{probabilidad * 100:.1f}".replace(".", ",")
    resultado = (
        f"Clase estimada: {clase} ({'positiva' if clase else 'negativa'}).\n"
        f"Probabilidad estimada de Outcome = 1: {porcentaje} %.\n"
        "Umbral de clasificación: 0,50.\n"
        f"Campos imputados con la mediana de entrenamiento: {', '.join(campos) or 'ninguno'}.\n\n"
        "La tabla muestra los aportes individuales al puntaje del modelo, "
        "respecto de los valores medios del entrenamiento. Un aporte positivo "
        "aumenta el puntaje de la clase 1 y uno negativo lo reduce. "
        "No son puntos porcentuales ni causas médicas.\n\n"
        "Prototipo académico. El resultado no es un diagnóstico ni una "
        "probabilidad clínicamente validada."
    )
    return resultado, tabla


def crear_app():
    resultados = (
        f"### Evaluación en el conjunto de prueba\n"
        f"Se reservaron {len(X_test)} registros. Exactitud: {exactitud:.1%}; "
        f"recall de la clase positiva: {sensibilidad:.1%}. "
        f"El modelo detectó {vp} de {vp + fn} positivos y dejó {fn} sin detectar. "
        f"También produjo {fp} falsos positivos.\n\n"
        "La población del dataset incluye mujeres de ascendencia Pima de 21 años "
        "o más. No se ha validado el uso del prototipo en otras poblaciones. "
        "La validación de entradas es básica y no verifica plausibilidad clínica."
    )
    return gr.Interface(
        fn=predecir_diabetes,
        inputs=[gr.Number(label=etiqueta) for etiqueta in ETIQUETAS],
        outputs=[
            gr.Textbox(label="Predicción y explicación", lines=12),
            gr.Dataframe(label="Aportes a esta predicción", interactive=False)
        ],
        examples=[datos.loc[0, VARIABLES].tolist(), datos.loc[1, VARIABLES].tolist()],
        title="Prototipo académico de clasificación de diabetes",
        description=(
            "Trabajo práctico de Ingeniería del Software · UGR. "
            "Ingresá las ocho variables o elegí un ejemplo. En glucosa, presión, "
            "pliegue, insulina e IMC, los ceros y campos vacíos se tratarán como "
            "datos faltantes. Cero embarazos se conserva como un valor válido."
        ),
        article=resultados,
        flagging_mode="never",
        analytics_enabled=False
    )


if __name__ == "__main__":
    app = crear_app()
    app.queue(default_concurrency_limit=1)
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        share=False
    )
