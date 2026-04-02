import asyncio
import os
import tempfile

import pygame

# Voz neuronal en español (México). Si Microsoft cambia el ID, ajusta aquí.
VOZ_NEURAL = "es-MX-JorgeNeural"


async def _generar_mp3_async(texto: str, ruta_mp3: str) -> None:
    import edge_tts

    communicate = edge_tts.Communicate(texto, VOZ_NEURAL)
    await communicate.save(ruta_mp3)


def hablar(texto: str) -> None:
    """Genera audio con edge-tts, reproduce con pygame y borra el temporal."""
    if not (texto or "").strip():
        return

    ruta_mp3 = None
    try:
        fd, ruta_mp3 = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)

        asyncio.run(_generar_mp3_async(texto, ruta_mp3))

        pygame.mixer.init()
        pygame.mixer.music.load(ruta_mp3)
        pygame.mixer.music.play()
        reloj = pygame.time.Clock()
        while pygame.mixer.music.get_busy():
            reloj.tick(10)

    except ImportError as exc:
        print(f"Falta una dependencia de audio: {exc}")
    except OSError as exc:
        print(f"Error de audio o red al generar/reproducir voz (¿sin internet?): {exc}")
    except Exception as exc:
        print(f"Error inesperado en salida de voz: {exc}")
    finally:
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass
        try:
            pygame.mixer.music.unload()
        except Exception:
            pass
        if ruta_mp3 and os.path.isfile(ruta_mp3):
            try:
                os.remove(ruta_mp3)
            except OSError:
                pass
