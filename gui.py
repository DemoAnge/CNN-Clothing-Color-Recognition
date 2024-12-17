from tkinter import Frame, Canvas, BOTH
from video import iniciar_video, cerrar_aplicacion

def create_main_frame(root):
    # Crear el frame principal
    frame_principal = Frame(root)
    frame_principal.pack(fill=BOTH, expand=True)

    # Crear el canvas para el video
    canvas_video = Canvas(frame_principal, bg="black")
    canvas_video.pack(fill=BOTH, expand=True)

    return frame_principal, canvas_video
