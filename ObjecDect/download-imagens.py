import urllib.request
import cv2
import numpy as np
import os

negativas_link = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n00483313'

#http://image-net.org/api/text/imagenet.synset.geturls?wnid=n03797390

headers = {}
headers['User-Agent'] = "Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17"
requisicao = urllib.request.Request(negativas_link, headers = headers)
resp = urllib.request.urlopen(requisicao)

respData = resp.read().decode()
numero_atual = 1

if not os.path.exists('neg'):
    os.makedirs('neg')

for i in respData.split('\n'):
    try:
        print(i)
        urllib.request.urlretrieve(i, "neg/" + str(numero_atual) + ".jpg")
        img = cv2.imread("neg/" + str(numero_atual) + ".jpg", cv2.IMREAD_GRAYSCALE)
        # deve ser maior que o tamanho da imagem positiva
        img_redimensionada = cv2.resize(img, (100, 100))
        cv2.imwrite("neg/" + str(numero_atual) + ".jpg", img_redimensionada)
        numero_atual += 1

    except Exception as e:
        print(str(e))

# https://pythonprogramming.net/haar-cascade-object-detection-python-opencv-tutorial/
