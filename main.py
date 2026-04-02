from jarvis_core import Jarvis
import time

def main():
    # Instanciamos la clase (Nace Jarvis)
    brain = Jarvis()
    
    # Primer aliento
    print("-------------------------------------------------")
    print(brain.activate())
    print("-------------------------------------------------")
    
    # Bucle de vida (Game Loop)
    while brain.is_active:
        try:
            # 1. INPUT (texto o voz, según el modo actual)
            user_input, input_error = brain.capturar_input_usuario()
            if input_error:
                print(f"JARVIS: {input_error}")
                continue
            if brain.input_mode == "voz":
                print(f"TU (voz): {user_input}")
            
            if user_input.lower() == "salir":
                print(f"\nJARVIS: {brain.shutdown()}")
                break
            
            # 2. SIMULACIÓN DE "PENSANDO" (Estética para tu tesis)
            # Esto le da realismo antes de que la IA responda
            print("Processing...", end="", flush=True)
            time.sleep(0.5) # Pequeña pausa dramática
            print("\r" + " " * 20 + "\r", end="") # Borra el "Processing..."
            
            # 3. PROCESAMIENTO
            response = brain.process_command(user_input)
            
            # 4. OUTPUT (Más tarde esto será: speaker.say(response))
            print(f"JARVIS: {response}")
            
        except KeyboardInterrupt:
            # Para salir elegantemente con Ctrl+C
            print(f"\n\nJARVIS: {brain.shutdown()}")
            break

if __name__ == "__main__":
    main()