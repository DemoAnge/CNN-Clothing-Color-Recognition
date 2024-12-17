import os
import cv2
import numpy as np
import threading
import time
from tkinter import Tk, Button, Label, Frame, Canvas, BOTH
from PIL import Image, ImageTk
from ultralytics import YOLO
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import signal
import sys

# Cargar dataset de colores
data = pd.read_csv('colores_mod.csv')
X = data[['Red', 'Green', 'Blue']]
y = data['Color']

# Definir el valor de K
K = 5

# Entrenar el modelo KNN
knn = KNeighborsClassifier(n_neighbors=K)
knn.fit(X, y)

# Función para predecir el color usando KNN
def predict_color(rgb):
    return knn.predict([rgb])[0]

# Cargar el modelo YOLOv8 entrenado
model = YOLO(r'yolo.pt')

# Variables de control
video_running = False
video_thread = None

def update_canvas(canvas, image):
    """Actualizar el canvas de tkinter con una nueva imagen"""
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img_tk = ImageTk.PhotoImage(image=img)
    canvas.create_image(0, 0, anchor="nw", image=img_tk)
    canvas.image = img_tk

def video_loop(canvas_video, canvas_top, label_top, canvas_low, label_low):
    cap = cv2.VideoCapture(0)
    global video_running

    while video_running:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocesar la imagen
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(source=img, save=False, show=False)

        labels_data = {'pecho': [], 'piernas': []}

        # Dibujar las cajas delimitadoras y etiquetas en el frame
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = model.names[cls]

                if conf > 0.5 and label in labels_data:
                    # Extraer la región de interés (ROI) y encontrar el color predominante
                    roi = frame[int(y1):int(y2), int(x1):int(x2)]
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    dominant_color = predict_color(roi_rgb.mean(axis=0).mean(axis=0))

                    # Guardar los datos de la etiqueta
                    labels_data[label].append({
                        'conf': conf,
                        'color': dominant_color,
                        'roi': roi
                    })

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label} {conf:.2f} Color: {dominant_color}', 
                                (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Update video canvas
        update_canvas(canvas_video, frame)

        # Update labels and miniatures for top and low
        if labels_data['pecho']:
            top_roi = labels_data['pecho'][0]['roi']  # Just take the first detected instance
            update_canvas(canvas_top, top_roi)
            top_conf = labels_data['pecho'][0]['conf']
            top_color = labels_data['pecho'][0]['color']
            label_top.config(text=f'Top (pecho): {top_conf*100:.2f}% Color: {top_color}', fg="green")

        if labels_data['piernas']:
            low_roi = labels_data['piernas'][0]['roi']  # Just take the first detected instance
            update_canvas(canvas_low, low_roi)
            low_conf = labels_data['piernas'][0]['conf']
            low_color = labels_data['piernas'][0]['color']
            label_low.config(text=f'Low (piernas): {low_conf*100:.2f}% Color: {low_color}', fg="red")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            video_running = False
            break

    cap.release()
    cv2.destroyAllWindows()

def start_video():
    global video_running, video_thread
    if not video_running:
        video_running = True
        video_thread = threading.Thread(target=video_loop, args=(canvas_video, canvas_top, label_top, canvas_low, label_low))
        video_thread.start()

def stop_video():
    global video_running
    video_running = False

def quit_app():
    stop_video()
    time.sleep(1)  # Esperar a que el hilo del video se detenga
    root.quit()  # Cerrar la ventana principal
    root.destroy()  # Destruir todos los widgets de la ventana principal
    os._exit(0)  # Terminar todo el proceso

# Crear la interfaz gráfica
root = Tk()
root.title("Control de Video")

# Crear el frame principal
main_frame = Frame(root)
main_frame.pack(fill=BOTH, expand=True)

# Crear los botones en la parte superior
button_frame = Frame(main_frame)
button_frame.pack(side="top", fill="x")

start_button = Button(button_frame, text="Iniciar Video", command=start_video)
start_button.pack(side="left")

stop_button = Button(button_frame, text="Detener Video", command=stop_video)
stop_button.pack(side="left")

quit_button = Button(button_frame, text="Salir", command=quit_app)
quit_button.pack(side="left")

# Crear el frame de video y las capturas laterales
video_frame = Frame(main_frame)
video_frame.pack(fill=BOTH, expand=True)

# Crear el canvas para el video
canvas_video = Canvas(video_frame, width=640, height=480)
canvas_video.pack(side="left", fill=BOTH, expand=True)

# Crear un frame para las etiquetas y sus miniaturas
labels_frame = Frame(video_frame)
labels_frame.pack(side="right", fill="y")

# Crear los canvases y labels para top y low en vertical
canvas_top = Canvas(labels_frame, width=160, height=120)
canvas_top.pack(side="top", fill="x")
label_top = Label(labels_frame, text="Top (pecho): ", fg="green")
label_top.pack(side="top")

canvas_low = Canvas(labels_frame, width=160, height=120)
canvas_low.pack(side="top", fill="x")
label_low = Label(labels_frame, text="Low (piernas): ", fg="red")
label_low.pack(side="top")

root.mainloop()
