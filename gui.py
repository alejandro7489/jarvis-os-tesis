import customtkinter as ctk
import threading
import os
from modules import ear
from main import Jarvis  # Cambia 'main' por el nombre de tu archivo sin el .py
mi_jarvis = Jarvis()
# Aquí importarás tu Jarvis real. Ejemplo:
# from jarvis_core import Jarvis
# mi_jarvis = Jarvis()

# Configuración visual de la ventana
ctk.set_appearance_mode("dark")  # Modo oscuro
ctk.set_default_color_theme("blue")  # Detalles en azul neón

class JarvisGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("JARVIS - AutoML Data Scientist")
        self.geometry("850x650")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Pantalla principal donde Jarvis y tú "chatean"
        self.textbox = ctk.CTkTextbox(self, font=("Consolas", 14), wrap="word")
        self.textbox.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="nsew")
        self.textbox.insert("0.0", "SISTEMA: Inicializando JARVIS... Módulos cargados.\nEsperando comandos...\n\n")
        self.textbox.configure(state="disabled")

        # Contenedor inferior para el input y botones
        self.input_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.input_frame.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="ew")
        self.input_frame.grid_columnconfigure(0, weight=1)

        # Caja de texto para escribir
        self.entry = ctk.CTkEntry(self.input_frame, height=40, font=("Consolas", 14), placeholder_text="Escribe un comando (ej. analizar datos)...")
        self.entry.grid(row=0, column=0, padx=(0, 10), sticky="ew")
        self.entry.bind("<Return>", self.enviar_texto) # Permite enviar con la tecla Enter

        # Botón de Enviar (Texto)
        self.btn_send = ctk.CTkButton(self.input_frame, text="Enviar", width=100, height=40, font=("Consolas", 14, "bold"), command=self.enviar_texto)
        self.btn_send.grid(row=0, column=1, padx=(0, 10))

        # Botón de Micrófono (Voz)
        self.btn_mic = ctk.CTkButton(self.input_frame, text="🎤 Hablar", width=100, height=40, fg_color="#b30000", hover_color="#800000", font=("Consolas", 14, "bold"), command=self.escuchar_voz)
        self.btn_mic.grid(row=0, column=2)

    def imprimir_mensaje(self, remitente, mensaje):
        """Función segura para escribir en la pantalla sin importar si viene de un hilo distinto"""
        self.textbox.configure(state="normal")
        self.textbox.insert("end", f"[{remitente}]: {mensaje}\n\n")
        self.textbox.see("end") # Hace scroll automático hacia abajo
        self.textbox.configure(state="disabled")

    def enviar_texto(self, event=None):
        comando = self.entry.get().strip()
        if not comando:
            return
        
        self.imprimir_mensaje("TÚ", comando)
        self.entry.delete(0, "end")
        
        # Procesamos el comando en segundo plano para no congelar la app
        threading.Thread(target=self.procesar_comando_jarvis, args=(comando,), daemon=True).start()

    def escuchar_voz(self):
        self.imprimir_mensaje("SISTEMA", "Ajustando micrófono... Habla ahora 🎤")
        self.btn_mic.configure(state="disabled") # Bloquea el botón temporalmente
        
        # Lanzamos el micrófono en un hilo paralelo para no congelar la ventana
        threading.Thread(target=self.hilo_escuchar_voz, daemon=True).start()
        
    def hilo_escuchar_voz(self):
        texto_capturado = ear.escuchar_comando()
        self.btn_mic.configure(state="normal") # Desbloquea el botón rojo
        
        if not texto_capturado:
            self.imprimir_mensaje("SISTEMA", "No detecté voz o hubo mucho ruido. Intenta de nuevo.")
            return
            
        if texto_capturado.startswith("error"):
            self.imprimir_mensaje("SISTEMA", f"Fallo en audio: {texto_capturado}")
            return
            
        # Si escuchó bien, lo muestra en pantalla y lo envía a Jarvis
        self.imprimir_mensaje("TÚ (Voz)", texto_capturado)
        self.procesar_comando_jarvis(texto_capturado)

    def procesar_comando_jarvis(self, comando):
        try:
            # Le pasamos el texto de la interfaz a tu motor Python
            # Ajusta "procesar_texto" al nombre real de la función que recibe comandos en tu código
            respuesta = mi_jarvis.process_command(comando) 
            
            # Si Jarvis devuelve texto, lo imprimimos en pantalla
            if respuesta:
                self.imprimir_mensaje("JARVIS", respuesta)
                
        except Exception as e:
            self.imprimir_mensaje("SISTEMA", f"Error interno: {str(e)}")

if __name__ == "__main__":
    app = JarvisGUI()
    app.mainloop()