import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# TP 3 - Cinco dados

# leer video
# achicar dimensiones quedándonos únicamente con las zonas de color verde
# filtrar por número de componentes conectadas
# encontrar frames quietos: comparar similaridad de stats, umbral de frames quietos consecutivos
# elegir un frame que chequeamos que esta quieto y realizar componentes conectadas para encontrar puntos
# grabar video agregando bounding box

videos = ['tirada_1.mp4', 'tirada_2.mp4', 'tirada_3.mp4', 'tirada_4.mp4']

for i, video in enumerate(videos):
    # Si no existe, crear el directorio 'tirada_i', donde i es el número de video.
    os.makedirs(f"tirada_{i+1}", exist_ok = True)
    # Leer video
    cap = cv2.VideoCapture(video) # Abre el archivo de video para su lectura.
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # Obtiene el ancho del video en píxeles.
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Obtiene la altura del video en píxeles.

    frame_num = 0
    while (cap.isOpened()): # Verifica si el video se abrió correctamente.
        ret, frame = cap.read() # 'ret' es True si la lectura fue exitosa y 'frame' es el contenido si la lectura fue exitosa.
        if ret:  
            frame = cv2.resize(frame, dsize=(int(width/3), int(height/3))) # Redimensiona el frame capturado.
            if frame_num == 0:
                # Convertir el frame a HSV para segmentar el color verde
                frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                #cv2.imshow('Frame HSV', frame_hsv)
                # Separamos los canales H, S y V para trabajar sobre H de manera más simple
                h, s, v = cv2.split(frame_hsv)
                #cv2.imshow('Frame canal H', h)            
                # Aplicar umbral al canal H para binarizar donde hay verde
                _, mask = cv2.threshold(h, 80, 255, cv2.THRESH_BINARY)
                #cv2.imshow('Frame binario umbralado "verde"', mask)
                # Componentes conectadas para encontrar el paño verde
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
                # Seleccionar área más grande, excluyendo el fondo
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                x, y, w, h, area = stats[largest_label]
            # Recortar región verde más grande
            frame_crop = frame[y:y+h-30, x+10:x+w-10]
            cv2.imshow('Frame recorte verde"', frame_crop)
            # Guardar el recorte en el archivo './tirada_i/frame_{frame_num}.jpg'.
            cv2.imwrite(os.path.join(f"tirada_{i+1}", f"{frame_num}.jpg"), frame_crop)
            frame_num += 1
            if cv2.waitKey(25) & 0xFF == ord('q'): # Espera 25 milisegundos a que se presione una tecla. Si se presiona 'q' se rompe el bucle y se cierra la ventana.
                break
        else:  
            break  
    cap.release() # Libera el objeto 'cap', cerrando el archivo.
    cv2.destroyAllWindows() # Cierra todas las ventanas abiertas.   

def hsv_bin(frame):
    # Convertir el frame a HSV
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    h, _, _ = cv2.split(frame_hsv)
    #cv2.imshow('Frame H', h)           
    # Aplicar umbral para binarizar la imagen
    frame_binario = cv2.inRange(h, 20, 100)
    frame_binario = cv2.bitwise_not(frame_binario)
    
    return frame_binario
    

def dados_quietos(directorio: str, bb_th=6, frames_th=5) -> dict:
    """
    Detecta el momento en que los dados están en reposo en cada directorio de frames.
    
    Args:
        directorio (str): Ruta del directorio que contiene los frames.
        bb_th (int): Umbral de diferencia máxima que puede haber entre las stats de cada frame para ser considerado quieto.
        frames_th (int): Frames consecutivos necesarios para confirmar reposo.

    Returns:
        dict: Diccionario con las rutas de los frames como clave y los frames como valor.
    """
    prev_bounding_boxes = list()
    frames_quietos_cont = 0
    frames_quietos = list()
    frames = sorted(os.listdir(directorio), key=lambda x: int(x.split('.')[0]))
    # Recorrer frames    
    for f in frames:
        path = os.path.join(directorio, f)
        # Leer la imagen
        frame = cv2.imread(path)
        frame_binario = hsv_bin(frame)
        cv2.imshow('Frame binario', frame_binario)
        # Filtrado por número de componentes conectadas == 6 (fondo + 5 dados)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(frame_binario)
        if num_labels == 6:
            bounding_boxes = stats[1:, :4]  # Excluir el fondo
            if len(prev_bounding_boxes) > 0:
                diferencias = np.abs(bounding_boxes - prev_bounding_boxes)
                max_diferencias = np.max(diferencias, axis=1)  # Máxima diferencia por bounding box
                if np.all(max_diferencias < bb_th):  # Si todas las diferencias están dentro de la tolerancia
                    frames_quietos_cont += 1
                else:
                    frames_quietos_cont = 0  # Reiniciar contador si hay movimiento
            else:
                frames_quietos_cont = 0  # Reset
            # Actualizar bounding boxes previas
            prev_bounding_boxes = bounding_boxes
            # Marcar frame como estático si cumple con el criterio
            if frames_quietos_cont >= frames_th:
                print(f"Dados quieto detectado en frame {f}")
                cv2.imshow('Frames quieto', frame)
                frames_quietos.append((path, frame))
        # Esperar a que se presione 'q' para salir
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break         
    cv2.destroyAllWindows()
    return frames_quietos


def detectar_puntos_dados(frame):
    """
    Detecta el número de cada dado.
    """
    puntos_dados = list()
    
    frame_binario = hsv_bin(frame)
    #cv2.imshow('Frame binario', frame_binario)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Encontrar dados
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(frame_binario)
    dados = list()
    # Iterar sobre labels 
    for i in range(1, num_labels):  # Empieza en 1 para saltar el fondo
        # Obtener bounding box de cada contorno
        x, y, w, h, _ = stats[i]
        # Recortar dado
        dado = frame[y:y+h, x:x+w]
        dados.append(dado)
        # Ver dado
        #plt.imshow(dado), plt.title(f"Dado {len(dados)}"), plt.axis('off'), plt.show()
       
    for dado in dados:
        puntos = 0
        dado_gray = cv2.cvtColor(dado, cv2.COLOR_RGB2GRAY)
        #plt.imshow(dado_gray, cmap='gray'), plt.title("Dado escala de grises"), plt.axis('off'), plt.show()
        # Umbralizamos para encontrar puntos
        _, dado_binary = cv2.threshold(dado_gray, 170, 255, cv2.THRESH_BINARY)
        #plt.imshow(dado_binary, cmap='gray'), plt.title("Dado binario"), plt.axis('off'), plt.show()
        # Encontrar puntos
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dado_binary)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > 5:
                puntos += 1
        puntos_dados.append(puntos)
        
    return puntos_dados  
    
    
def grabar_video(directorio: str, frames_quietos: list, puntos_dados: list, output: str):
    frames = sorted(os.listdir(directorio), key=lambda x: int(x.split('.')[0]))
    first_frame = cv2.imread(os.path.join(directorio, frames[0]))
    height, width, _ = first_frame.shape
    out = cv2.VideoWriter(f'{output}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
    for f in frames:
        path = os.path.join(directorio, f)
        frame = cv2.imread(path)
        if path in [f[0] for f in frames_quietos]:
            frame_binario = hsv_bin(frame)     
            # Detectar bounding boxes
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(frame_binario)
            for i in range(1, num_labels):  # Saltar el fondo
                x, y, w, h, _ = stats[i]
                # Dibujar bounding box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Agregar etiqueta con puntos
                cv2.putText(frame, f"{puntos_dados[i-1]}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        # Escribir el frame en el video
        out.write(frame)
        cv2.imshow('Frames con etiquetas', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    out.release()
    cv2.destroyAllWindows()
    print(f"Video guardado como: {output}")
    
# Lista de directorios creados
directorios = [d for d in os.listdir('.') if os.path.isdir(d)]
# Procesar cada carpeta de frames
for directorio in directorios:
    frames_quietos = dados_quietos(directorio)
    puntos_dados = detectar_puntos_dados(frames_quietos[-1][1])
    grabar_video(directorio, frames_quietos, puntos_dados, f'{directorio}_output')
    