import cv2
import threading
from PIL import Image, ImageTk
from knn import predecir_color_KNN
from yolo import model
import time
import os

# Variables de control
video_en_ejecucion = True

def actualizar_canvas(canvas, imagen):
    """Actualizar el canvas de tkinter con una nueva imagen"""
    ancho_canvas = canvas.winfo_width()
    alto_canvas = canvas.winfo_height()
    img = cv2.resize(imagen, (ancho_canvas, alto_canvas))  # Redimensionar la imagen al tamaño del canvas
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convertir la imagen a RGB
    img_tk = ImageTk.PhotoImage(image=img)
    canvas.create_image(0, 0, anchor="nw", image=img_tk)  # Dibujar la imagen en el canvas
    canvas.image = img_tk  # Guardar una referencia a la imagen para evitar que se recoja como basura

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
                    color_predominante = predecir_color_KNN(roi_rgb.mean(axis=0).mean(axis=0))

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

def iniciar_video(canvas_video):
    global hilo_video
    hilo_video = threading.Thread(target=bucle_video, args=(canvas_video,))
    hilo_video.start()

def cerrar_aplicacion(root):
    global video_en_ejecucion
    video_en_ejecucion = False
    time.sleep(1)  # Esperar a que el hilo del video se detenga
    root.quit()  # Cerrar la ventana principal
    root.destroy()  # Destruir todos los widgets de la ventana principal
    os._exit(0)  # Terminar todo el proceso
