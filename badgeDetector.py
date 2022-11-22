import cv2                                          # lib cv2 para detecção de imagem
import numpy as np                                  # lib numpy para converter dados em arrays e integers
import pyzbar.pyzbar as pyzbar                      # lib para detectar e ler qrcode

def badgeDetector(frame):

    # convertendo os bits do frame de rgb(bgr) para hsv
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lowerGreen = np.array([52, 255, 164]) # Cor Verde
    upperGreen = np.array([255, 255, 255]) # Limite
    mask = cv2.inRange(hsv, lowerGreen, upperGreen) # Máscara para detecção da cor
    result = cv2.bitwise_and(frame, frame, mask=mask) # Aplica a máscara frame por frame do vídeo
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) # Converte a imagem para escala de cinza
    _, borda = cv2.threshold(gray, 3, 255, cv2.THRESH_BINARY) # Binarização da imagem e a variável borda conténdo a imagem com cores invertidas
    contornos, _ = cv2.findContours(borda, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # detecta diferenciação na cor da imagem e marca como contorno
    
    for contorno in contornos:
        decodedObjects = pyzbar.decode(frame) # Lê e decodifica qualquer QRcode detectado dentro dos frames do vídeo
        area = cv2.contourArea(contorno) # computa a área de contorno dentro da lista da variável "contornos"
        if area > 200:
            # Se não for detectado nenhum QRcode dentro de decodedObjects, subtrair da variável "tries"
            if decodedObjects:
                dataString = decodedObjects[0].data.decode("utf-8")
                if dataString == 'Marcos':
                    return True
                else:
                    return False