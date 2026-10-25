# Prototipo académico de clasificación de diabetes

Aplicación Gradio para el trabajo práctico de Ingeniería del Software (UGR).
Incluye la probabilidad estimada, la clase predicha, los aportes individuales
al puntaje logístico y la identificación de mediciones imputadas.

## Archivos

- `app.py`: carga, entrenamiento y aplicación completa.
- `requirements.txt`: versiones de dependencias directas.
- `diabetes.csv`: los 768 registros utilizados en el Colab, con encabezados.
- `.python-version`: selecciona Python 3.12.
- `render.yaml`: configuración opcional para desplegar mediante Render Blueprint.

## Ejecutar en la computadora

Requiere Python 3.12. Desde esta carpeta:

```bash
python -m venv .venv
```

Activar el entorno en Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

En Linux o macOS:

```bash
source .venv/bin/activate
```

Luego:

```bash
python -m pip install -r requirements.txt
python app.py
```

Abrir http://localhost:7860. El servidor no crea un túnel público de Gradio.

## Publicar en Render desde GitHub

1. Crear un repositorio y subir los archivos descomprimidos a su raíz.
   No subir solamente el ZIP. Incluir `.python-version`.
2. En Render elegir **New > Web Service** y conectar el repositorio.
3. Seleccionar lenguaje **Python 3** y plan **Free**.
4. Configurar **Build Command**: `pip install -r requirements.txt`.
5. Configurar **Start Command**: `python app.py`.
6. Dejar **Root Directory** vacío si los archivos están en la raíz.
   Si se subió la carpeta completa como subcarpeta, indicar `diabetes_render`.
7. Crear el servicio y esperar a que finalice el despliegue.
8. Abrir la URL que devuelve Render, probar ambos ejemplos y una entrada inválida.
9. Copiar ese enlace al Colab e incorporar capturas reales del formulario y del resultado.

Alternativa: usar **New > Blueprint** con el repositorio; Render leerá `render.yaml`.
El archivo YAML no publica por sí solo la aplicación.

Render entrega el puerto por la variable `PORT`; `app.py` la utiliza y escucha
en `0.0.0.0`. No se usa `share=True` en este alojamiento.
El plan gratuito puede suspender el servicio por inactividad y demorar al reactivarlo.

## Relación con el Colab

El modelo se entrena una vez al iniciar cada proceso, con el CSV local. No depende
de Drive, Colab ni de descargas externas del dataset durante el arranque.
La separación es 80/20, estratificada, con `random_state=42`: 614 filas para
entrenamiento y 154 para prueba. Se conserva esta separación para que los resultados
sean comparables con la notebook; no se reentrena con los 768 registros completos.

El pipeline aplica en entrenamiento e inferencia:

1. Ceros de Glucose, BloodPressure, SkinThickness, Insulin y BMI a NaN.
2. Imputación por mediana aprendida solamente en entrenamiento.
3. Estandarización aprendida solamente en entrenamiento.
4. Regresión logística (`max_iter=1000`).

La probabilidad es la estimación de `Outcome=1`; la clase se obtiene con `predict`.
Los aportes son `coeficiente × valor estandarizado`. Se muestran respecto de los
valores medios de entrenamiento y se ordenan por magnitud absoluta. No son porcentajes,
factores causales ni efectos de una intervención. Se omite el intercepto de la tabla;
la probabilidad sí lo incluye. Se informa cuando una variable fue imputada.

## Procedencia y alcance de los datos

Fuente original identificada en la documentación: National Institute of Diabetes
and Digestive and Kidney Diseases (NIDDK). Dataset Pima Indians Diabetes.
Población: mujeres de ascendencia Pima, de 21 años o más. El CSV incluido es la copia
de trabajo utilizada en la carpeta del TP.

- Datos usados en la notebook: https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
- Descripción: https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.names

Demostración académica: no constituye un diagnóstico y sus probabilidades no están
validadas como riesgo clínico. No se ha validado su uso en otras poblaciones.
La aplicación no implementa guardado de consultas ni solicita datos identificatorios.

## Documentación del alojamiento

- https://render.com/docs/web-services
- https://render.com/docs/python-version
- https://render.com/docs/free
