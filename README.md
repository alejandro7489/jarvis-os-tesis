# 🤖 JARVIS: Autonomous AutoML & Engineering Agent
> **Proyecto para la Hackathon Jebi 2026 - Arequipa, Perú**

**JARVIS** (Just A Rather Very Intelligent System) es un agente multimodal de baja latencia diseñado para democratizar el acceso al **Deep Learning** y la **Ciencia de Datos**. A diferencia de los modelos de lenguaje tradicionales, JARVIS no solo conversa; razona sobre datasets, genera código de entrenamiento dinámico y optimiza modelos de forma autónoma.

## 🚀 Key Features

* **⚡ Ultra-Low Latency Inference:** Integración con **Llama 3.1** a través de la infraestructura de **Groq (LPUs)** para respuestas casi instantáneas.
* **🧠 Hybrid Evolutionary AutoML:** Identificación automática de tareas de **Clasificación** y **Regresión**. Implementa **GridSearchCV**, **SMOTE** y **Transformaciones Log1p** para maximizar la precisión.
* **🎙️ Neural Voice Interface:** Comunicación natural mediante voces neuronales de **Edge-TTS**, permitiendo una interacción fluida sin manos.
* **📊 Explainable AI (XAI):** Capacidad de justificar predicciones basadas en la importancia de características (*Feature Importance*), eliminando el sesgo de "caja negra".
* **🖥️ Modern GUI:** Interfaz desarrollada en **CustomTkinter** con soporte para simulaciones de casos en tiempo real y visualización de gráficos dinámicos.

## 🛠️ Tech Stack

* **Language:** Python 3.12
* **AI Engine:** Llama 3.1 (Groq API) & Ollama (Local fallback)
* **Data Science:** Pandas, Scikit-Learn, XGBoost, Imbalanced-Learn
* **Voice:** Edge-TTS, SpeechRecognition & Pygame
* **UI:** CustomTkinter & Matplotlib (TkAgg)
* **Environment:** Dotenv para gestión segura de credenciales

## 📈 Benchmarking (Medical Insurance Case)

| Métrica | Valor Inicial | Valor Optimizado | Impacto |
| :--- | :--- | :--- | :--- |
| **R² Score** | `0.8569` | **0.8715** | +1.46% de varianza explicada |
| **MAE** | `$2,646.39` | **$2,232.76** | **Ahorro de error: $413.63** |
| **Strategy** | Base Regressor | **Log1p Transformation** | Estabilización de outliers |

## 📂 Project Structure

```text
jarvis_project/
├── modules/
│   ├── voice_output.py    # Generación de voz neuronal (Edge-TTS)
│   ├── data_scientist.py  # Pipeline de AutoML, GridSearch y Optimización
│   ├── data_viz.py        # Visualización de gráficos (Scatter Plots, Residuos)
│   └── model_trainer.py   # Entrenamiento y utilidades de predicción
├── gui.py                 # Interfaz gráfica moderna (CustomTkinter)
├── jarvis_core.py         # Núcleo de razonamiento y lógica del agente
├── main.py                # Punto de entrada de la aplicación
├── requirements.txt       # Dependencias del sistema
├── .env                   # Variables de entorno (Secrets)
└── .gitignore             # Protección de archivos sensibles (Ignora .env)

Desarrollado por: Fernando Alejandro Lizárraga Barrios

Universidad La Salle - Arequipa
