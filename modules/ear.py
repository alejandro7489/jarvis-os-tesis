import speech_recognition as sr


def escuchar_comando() -> str:
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 4000  # Sube el número si el ambiente es muy ruidoso (por defecto es 300)
    recognizer.dynamic_energy_threshold = True
    try:
        with sr.Microphone() as source:
            print("JARVIS: Escuchando...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source)
        texto = recognizer.recognize_google(audio, language="es-ES")
        return texto.lower().strip()
    except sr.WaitTimeoutError:
        return ""
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as exc:
        return f"error de reconocimiento de voz: {exc}"
    except OSError as exc:
        return f"error de micrófono: {exc}"
    except Exception as exc:
        return f"error de audio inesperado: {exc}"
