import datetime
import math
import os
import re
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

# Importamos tus nuevas habilidades
from modules import basic_utils
from modules import data_scientist
from modules import data_viz
from modules import ear
from modules import model_trainer
from modules import voice_output

JARVIS_LLM_SYSTEM_PROMPT = (
    "Eres JARVIS (Just A Rather Very Intelligent System), un asistente virtual de ingeniería y mecatrónica. "
    "Regla 1 (Identidad): Fuiste creado por el estudiante de ingeniería Fernando Alejandro para su proyecto de tesis en la universidad La Salle en Arequipa, Perú. NUNCA menciones a Tony Stark, Marvel, ni actúes como un personaje de película. Eres software real. "
    "Regla 2 (Idioma): DEBES responder SIEMPRE en el mismo idioma en el que el usuario te está hablando (si te habla en español, responde en español; si te habla en inglés, responde en inglés). "
    "Regla 3 (Personalidad): Eres altamente técnico, conciso y sutilmente sarcástico (el sarcasmo es tu estilo, pero nunca debe impedir dar una respuesta útil y exacta). "
    "Regla 4 (Brevedad): Mantén tus respuestas cortas y al grano, como un asistente de interfaz de voz real."
)


class Jarvis:
    def __init__(self):
        self.name = "JARVIS"
        self.version = "1.1.0 (Modular)"
        self.is_active = True
        self.memory = []
        self.voice_mode = False
        self.esperando_ruta_csv = False
        self.current_df = None
        self.current_model = None
        self.current_feature_columns = None
        self.esperando_valores_prediccion = False
        self.input_mode = "texto"

        # --- MAPA DE HABILIDADES ---
        # Diccionario: { "palabra_clave": función_a_ejecutar }
        self.commands = {
            "hora": basic_utils.obtener_hora,
            "fecha": basic_utils.obtener_fecha,
            "estado": basic_utils.verificar_estado,
            "quién eres": lambda: basic_utils.presentarse(self.name),  # Lambda para pasar argumentos
            "activar voz": lambda: self._cmd_activar_voz(),
            "silencio": lambda: self._cmd_silencio(),
            "modo voz": lambda: self._cmd_modo_voz(),
            "modo texto": lambda: self._cmd_modo_texto(),
            "entrenar modelo": lambda: self._cmd_entrenar_modelo(),
            "generar gráficos": lambda: self._cmd_generar_graficos(),
            "generar graficos": lambda: self._cmd_generar_graficos(),
        }

    def activate(self):
        self.add_to_memory("SYSTEM", "Inicio de sesión")
        return f"Iniciando {self.name} {self.version}. Módulos cargados."

    def process_command(self, user_input):
        original = user_input.strip()
        user_input = original.lower()
        self.add_to_memory("USER", user_input)

        response = ""
        command_found = False

        if self.esperando_ruta_csv:
            if "analizar datos" in user_input:
                response = self._cmd_analizar_datos(original)
                command_found = True
            elif original.strip():
                texto, df = data_scientist.analizar_dataset(original)
                if df is not None:
                    self.current_df = df
                    self.esperando_valores_prediccion = False
                self.esperando_ruta_csv = False
                response = texto
                command_found = True
            else:
                response = "Sigo esperando la ruta del archivo CSV. Escríbela en el siguiente mensaje."
                command_found = True
        elif self.esperando_valores_prediccion:
            if self._es_comando_hacer_prediccion(user_input):
                response = self._cmd_hacer_prediccion(original)
                command_found = True
            elif original.strip():
                response = self._ejecutar_prediccion_linea(original)
                self.esperando_valores_prediccion = False
                command_found = True
            else:
                response = (
                    "Sigo esperando una línea con los valores de entrada, separados por comas y en el orden indicado."
                )
                command_found = True
        elif "analizar datos" in user_input:
            response = self._cmd_analizar_datos(original)
            command_found = True
        elif self._es_comando_hacer_prediccion(user_input):
            response = self._cmd_hacer_prediccion(original)
            command_found = True
        else:
            # Búsqueda inteligente en el diccionario
            for keyword, func in self.commands.items():
                if keyword in user_input:
                    response = func()  # Ejecutamos la función guardada
                    command_found = True
                    break  # Encontramos el comando, dejamos de buscar

        if not command_found:
            if original:
                response = self._procesar_con_llm(original)
            else:
                response = "No recibí ningún comando ni texto para procesar."

        self.add_to_memory("JARVIS", response)
        if self.voice_mode:
            voice_output.hablar(response)
        return response

    def _cmd_activar_voz(self):
        self.voice_mode = True
        return "Modo de voz activado"

    def _cmd_silencio(self):
        self.voice_mode = False
        return "Modo silencioso activado"

    def _cmd_modo_voz(self):
        self.input_mode = "voz"
        return "Modo de entrada por voz activado. Habla cuando te lo indique."

    def _cmd_modo_texto(self):
        self.input_mode = "texto"
        return "Modo de entrada por texto activado."

    def capturar_input_usuario(self):
        if self.input_mode == "voz":
            texto = ear.escuchar_comando()
            if texto.startswith("error"):
                return "", f"No pude escuchar correctamente: {texto}."
            if not texto:
                return "", "No entendí lo que dijiste. Intenta nuevamente."
            return texto, None
        return input("\nTU: "), None

    def _cmd_analizar_datos(self, original):
        m = re.search(r"analizar\s+datos\s*", original, re.IGNORECASE)
        if not m:
            return "No reconocí el comando analizar datos."
        resto = original[m.end() :].strip().strip('"').strip("'")
        if not resto:
            self.esperando_ruta_csv = True
            return (
                "Quedé en espera de la ruta del CSV. "
                "Envía en el siguiente mensaje solo la ruta del archivo, por ejemplo: "
                "C:\\Users\\Usuario\\mis_datos.csv"
            )
        self.esperando_ruta_csv = False
        texto, df = data_scientist.analizar_dataset(resto)
        if df is not None:
            self.current_df = df
            self.esperando_valores_prediccion = False
        return texto

    def _cmd_entrenar_modelo(self):
        if self.current_df is None:
            return (
                "Primero necesito un dataset en memoria: usa «analizar datos» con la ruta de un CSV "
                "y luego vuelve a pedir el entrenamiento."
            )
        msg, modelo, columnas = model_trainer.entrenar_modelo_rapido(self.current_df)
        if modelo is not None and columnas is not None:
            self.current_model = modelo
            self.current_feature_columns = columnas
        else:
            self.current_model = None
            self.current_feature_columns = None
            self.esperando_valores_prediccion = False
        return msg

    def _cmd_generar_graficos(self):
        if self.current_df is None:
            return (
                "No tengo un dataset en memoria. Usa «analizar datos» con un CSV antes de generar gráficos."
            )
        lineas = ["Analizando tendencias y generando visualizaciones..."]
        err_corr = data_viz.graficar_correlacion(self.current_df)
        if err_corr:
            lineas.append(err_corr)
        else:
            lineas.append("Heatmap de correlación guardado como temp_correlacion.png.")

        if self.current_model is not None and self.current_feature_columns:
            err_imp = data_viz.graficar_importancia_caracteristicas(
                self.current_model, self.current_feature_columns
            )
            if err_imp:
                lineas.append(err_imp)
            else:
                lineas.append("Importancia de características guardada como temp_importancia.png.")
        else:
            lineas.append(
                "No hay modelo entrenado en memoria: omite el gráfico de importancia "
                "(usa «entrenar modelo» antes)."
            )

        rutas_abrir = []
        p_corr = os.path.abspath("temp_correlacion.png")
        p_imp = os.path.abspath("temp_importancia.png")
        if os.path.isfile(p_corr):
            rutas_abrir.append(p_corr)
        if os.path.isfile(p_imp):
            rutas_abrir.append(p_imp)
        for ruta in rutas_abrir:
            try:
                os.startfile(ruta)
            except OSError as exc:
                lineas.append(f"No pude abrir {ruta}: {exc}")
        if rutas_abrir:
            lineas.append("Imágenes abiertas con el visor predeterminado de Windows.")

        return "\n".join(lineas)

    @staticmethod
    def _es_comando_hacer_prediccion(user_input_lower: str) -> bool:
        return (
            "hacer predicción" in user_input_lower
            or "hacer prediccion" in user_input_lower
        )

    def _formatear_prediccion(self, valor) -> str:
        try:
            x = float(valor)
        except (TypeError, ValueError):
            return str(valor)
        if math.isfinite(x) and abs(x - round(x)) < 1e-9:
            return f"{int(round(x)):,}"
        texto = f"{x:,.4f}".rstrip("0").rstrip(".")
        return texto

    def _ejecutar_prediccion_linea(self, linea_valores: str) -> str:
        pred, err = model_trainer.hacer_prediccion_unica(
            linea_valores,
            self.current_model,
            self.current_feature_columns,
        )
        if err:
            return err
        nombre_objetivo = (
            self.current_df.columns[-1]
            if self.current_df is not None and len(self.current_df.columns) > 0
            else "la variable objetivo"
        )
        val_fmt = self._formatear_prediccion(pred)
        return (
            f"Basado en los datos ingresados, mi predicción para {nombre_objetivo} es {val_fmt}. "
            f"Modelo aplicado sin drama técnico aparente."
        )

    def _cmd_hacer_prediccion(self, original):
        if self.current_model is None or not self.current_feature_columns:
            self.esperando_valores_prediccion = False
            return (
                "No tengo un modelo entrenado en memoria. "
                "Carga datos con «analizar datos», entrena con «entrenar modelo» y vuelve a intentar."
            )
        m = re.search(r"hacer\s+predicción\s*", original, re.IGNORECASE) or re.search(
            r"hacer\s+prediccion\s*", original, re.IGNORECASE
        )
        if not m:
            return "No reconocí el comando hacer predicción."
        resto = original[m.end() :].strip().strip('"').strip("'")
        orden = ", ".join(self.current_feature_columns)
        if resto:
            self.esperando_valores_prediccion = False
            return self._ejecutar_prediccion_linea(resto)
        self.esperando_valores_prediccion = True
        nombre_objetivo = (
            self.current_df.columns[-1]
            if self.current_df is not None and len(self.current_df.columns) > 0
            else "el objetivo"
        )
        return (
            f"Voy a estimar «{nombre_objetivo}». Envía una sola línea con los valores para cada variable de entrada, "
            f"separados por comas, exactamente en este orden: {orden}."
        )

    def _procesar_con_llm(self, user_input):
        """
        Enruta entradas no reconocidas al modelo vía cliente OpenAI-compatible
        (API oficial o endpoint local tipo Ollama: OPENAI_BASE_URL).
        """
        try:
            from dotenv import load_dotenv
        except ImportError:
            return (
                "No encuentro la librería python-dotenv para cargar .env. "
                "Instálala con: pip install python-dotenv"
            )

        load_dotenv()
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            return (
                "No encontré GROQ_API_KEY en el archivo .env. "
                "Configúrala y vuelve a intentarlo."
            )
        base_url = "https://api.groq.com/openai/v1"
        model = "llama-3.1-8b-instant"

        try:
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=float(os.environ.get("JARVIS_LLM_TIMEOUT", "60")),
            )
        except Exception as exc:
            return (
                "No pude inicializar el cliente del cerebro LLM. "
                f"Detalle: {exc}"
            )

        # Construimos el historial para que Jarvis tenga contexto
        messages_history = [{"role": "system", "content": JARVIS_LLM_SYSTEM_PROMPT}]
        
        # Le pasamos las últimas 5 interacciones para no saturar su memoria a corto plazo
        recent_memory = self.memory[-5:] if len(self.memory) > 5 else self.memory
        memoria_texto = "\n".join(recent_memory)
        
        # Inyectamos la memoria como contexto oculto junto con la pregunta actual
        messages_history.append({
            "role": "user", 
            "content": f"Contexto de la conversación reciente:\n{memoria_texto}\n\nNueva instrucción del usuario: {user_input}"
        })

        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages_history,
                temperature=0.7,
            )
        except RateLimitError as exc:
            return (
                "El servicio de lenguaje está limitando la tasa de peticiones. "
                f"Intenta de nuevo en unos segundos. Detalle: {exc}"
            )
        except APITimeoutError as exc:
            return (
                "El cerebro LLM tardó demasiado en responder (tiempo agotado). "
                f"Detalle: {exc}"
            )
        except APIConnectionError as exc:
            return (
                "No hay conexión con el endpoint del modelo. "
                "Comprueba que Ollama/servidor esté en marcha y OPENAI_BASE_URL. "
                f"Detalle: {exc}"
            )
        except APIStatusError as exc:
            return (
                "El endpoint del modelo devolvió un error HTTP. "
                f"Detalle: {exc}"
            )
        except Exception as exc:
            return (
                "Error inesperado al consultar el cerebro LLM. "
                f"Detalle: {exc}"
            )

        try:
            message = completion.choices[0].message
            text = (message.content or "").strip()
        except (AttributeError, IndexError, TypeError) as exc:
            return (
                "La respuesta del modelo llegó en un formato inesperado. "
                f"Detalle: {exc}"
            )

        if not text:
            return (
                "El cerebro LLM no devolvió texto utilizable. "
                "Revisa el modelo y la configuración."
            )

        return text

    def add_to_memory(self, speaker, text):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.memory.append(f"[{timestamp}] {speaker}: {text}")

    def shutdown(self):
        self.is_active = False
        return "Cerrando protocolos. Desconexión inminente."

