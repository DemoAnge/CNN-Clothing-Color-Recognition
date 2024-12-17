from tkinter import Tk
from gui import create_main_frame, iniciar_video, cerrar_aplicacion

# Crear la interfaz gráfica
root = Tk()
root.title("Reconocimiento de colores")

# Crear el frame principal
frame_principal, canvas_video = create_main_frame(root)

root.protocol("WM_DELETE_WINDOW", lambda: cerrar_aplicacion(root))
root.bind('<Escape>', lambda e: cerrar_aplicacion(root))  # Vincular la tecla Esc para cerrar la aplicación

# Iniciar el video automáticamente al abrir la aplicación
iniciar_video(canvas_video)

# Inicializar la ventana principal
root.mainloop()
