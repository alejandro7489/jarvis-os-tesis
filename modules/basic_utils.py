import datetime
import locale

# Intentamos configurar español para las fechas (funciona en la mayoría de Windows/Linux)
try:
    locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8') 
except:
    try:
        locale.setlocale(locale.LC_TIME, 'Spanish') # Para Windows a veces
    except:
        pass # Si falla, se queda en inglés, no pasa nada

def obtener_hora():
    now = datetime.datetime.now()
    return f"Son las {now.strftime('%H:%M')}."

def obtener_fecha():
    now = datetime.datetime.now()
    return f"Hoy es {now.strftime('%A %d de %B de %Y')}."

def verificar_estado():
    return "Energía estable. Todos los sistemas nominales, señor."

def presentarse(nombre):
    return f"Soy {nombre}, una inteligencia artificial diseñada para asistencia técnica y táctica."