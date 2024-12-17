import os
import cv2  # OpenCV para procesamiento de imágenes
import numpy as np  # Numpy para cálculos numéricos
import threading  # Para manejo de hilos
import time  # Para manejo de tiempo
from tkinter import Tk, Frame, Canvas, BOTH  # Tkinter para la interfaz gráfica
from PIL import Image, ImageTk  # PIL para manejar imágenes
from ultralytics import YOLO  # YOLO para detección de objetos
import pandas as pd  # Pandas para manejo de datos

# Cargar el conjunto de datos de colores desde un archivo CSV
data = pd.read_csv('colores_mod.csv')
X = data[['Red', 'Green', 'Blue']].values  # Obtener los valores RGB
y = data['Color'].values  # Obtener los nombres de los colores

# Definir el valor de K para KNN
K = 5

# Función para calcular la distancia euclidiana entre dos puntos
def distancia_euclidiana(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Función para predecir el color usando KNN manual para una pequeña zona central
def predecir_color_KNN_central(roi_rgb):
    alto, ancho, _ = roi_rgb.shape
    centro_y, centro_x = alto // 2, ancho // 2
    tam_cuadro = 2  # Tamaño del área central a considerar
    roi_central = roi_rgb[centro_y - tam_cuadro:centro_y + tam_cuadro, centro_x - tam_cuadro:centro_x + tam_cuadro]

    # Aplanar la ROI central a una lista de colores
    colores = roi_central.reshape((-1, 3))

    # Calcular las distancias para cada color en la ROI central
    predicciones = []
    for color in colores:
        distancias = []
        for i in range(len(X)):
            distancia = distancia_euclidiana(X[i], color)
            distancias.append((distancia, y[i]))
        distancias.sort(key=lambda x: x[0])
        vecinos = distancias[:K]

        # Contar las ocurrencias de cada color en los K vecinos más cercanos
        conteos = {}
        for _, etiqueta in vecinos:
            if etiqueta in conteos:
                conteos[etiqueta] += 1
            else:
                conteos[etiqueta] = 1

        # Devolver el color con más ocurrencias
        color_predominante = max(conteos, key=conteos.get)
        predicciones.append(color_predominante)

    # Determinar el color más frecuente en la ROI central
    from collections import Counter
    color_final = Counter(predicciones).most_common(1)[0][0]

    return color_final

# Cargar el modelo YOLOv8 entrenado
model = YOLO(r'yolo.pt')

# Variables de control
video_en_ejecucion = True

# Función para actualizar el canvas de tkinter con una nueva imagen
def actualizar_canvas(canvas, imagen):
    ancho_canvas = canvas.winfo_width()
    alto_canvas = canvas.winfo_height()
    img = cv2.resize(imagen, (ancho_canvas, alto_canvas))  # Redimensionar la imagen al tamaño del canvas
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convertir la imagen a RGB
    img_tk = ImageTk.PhotoImage(image=img)
    canvas.create_image(0, 0, anchor="nw", image=img_tk)  # Dibujar la imagen en el canvas
    canvas.image = img_tk  # Guardar una referencia a la imagen para evitar que se recoja como basura

# Función principal del bucle de video
def bucle_video(canvas_video):
    cap = cv2.VideoCapture(0)  # Abrir la cámara
    global video_en_ejecucion

    while video_en_ejecucion:
        ret, frame = cap.read()  # Leer un frame de la cámara
        if not ret:
            break

        # Convertir el frame a RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Realizar la predicción usando YOLO
        resultados = model.predict(source=img, save=False, show=False)

        etiquetas_datos = {'pecho': [], 'piernas': []}

        # Dibujar las cajas delimitadoras y etiquetas en el frame
        for resultado in resultados:
            cajas = resultado.boxes
            for caja in cajas:
                x1, y1, x2, y2 = caja.xyxy[0].tolist()  # Coordenadas de la caja
                conf = caja.conf[0].item()  # Confianza de la predicción
                cls = int(caja.cls[0].item())  # Clase de la predicción
                etiqueta = model.names[cls]  # Nombre de la clase

                # Si la etiqueta es 'persona', dibujar el recuadro sin color
                if conf > 0.5 and etiqueta == 'persona':
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    texto = f'{etiqueta} {conf:.2f}%'
                    (ancho_texto, alto_texto), baseline = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (int(x1), int(y1) - alto_texto - 10), (int(x1) + ancho_texto, int(y1)), (0, 0, 0), -1)
                    cv2.putText(frame, texto, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Filtrar solo las clases 'pecho' y 'piernas' para el reconocimiento de color
                elif conf > 0.5 and etiqueta in etiquetas_datos:
                    # Extraer la región de interés (ROI) y encontrar el color predominante
                    roi = frame[int(y1):int(y2), int(x1):int(x2)]
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    color_predominante = predecir_color_KNN_central(roi_rgb)

                    # Dibujar la caja delimitadora
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # Preparar el texto y el recuadro
                    texto = f'{etiqueta} {conf:.2f}% - {color_predominante}'
                    (ancho_texto, alto_texto), baseline = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (int(x1), int(y1) - alto_texto - 10), (int(x1) + ancho_texto, int(y1)), (0, 0, 0), -1)
                    
                    # Dibujar el texto
                    cv2.putText(frame, texto, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Actualizar el canvas del video
        actualizar_canvas(canvas_video, frame)

    cap.release()  # Liberar la cámara
    cv2.destroyAllWindows()  # Cerrar todas las ventanas de OpenCV

# Función para iniciar el video en un nuevo hilo
def iniciar_video(canvas_video):
    global hilo_video
    hilo_video = threading.Thread(target=bucle_video, args=(canvas_video,))
    hilo_video.start()

# Función para cerrar la aplicación
def cerrar_aplicacion(root):
    global video_en_ejecucion
    video_en_ejecucion = False
    time.sleep(1)  # Esperar a que el hilo del video se detenga
    root.quit()  # Cerrar la ventana principal
    root.destroy()  # Destruir todos los widgets de la ventana principal
    os._exit(0)  # Terminar todo el proceso

# Crear la interfaz gráfica
root = Tk()
root.title("Reconocimiento de colores")

# Crear el frame principal
frame_principal = Frame(root)
frame_principal.pack(fill=BOTH, expand=True)

# Crear el canvas para el video
canvas_video = Canvas(frame_principal, bg="black")
canvas_video.pack(fill=BOTH, expand=True)

root.protocol("WM_DELETE_WINDOW", lambda: cerrar_aplicacion(root))
root.bind('<Escape>', lambda e: cerrar_aplicacion(root))  # Vincular la tecla Esc para cerrar la aplicación

# Iniciar el video automáticamente al abrir la aplicación
iniciar_video(canvas_video)

# Inicializar la ventana principal
root.mainloop()
