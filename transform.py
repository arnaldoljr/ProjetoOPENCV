import numpy as np
import cv2

img = cv2.imread('ceara.png',0)
cols = img.shape[1]
lin = img.shape[0]


''' DESLOCAMENTO DE UMA IMAGEM
M = np.float32([[1,0,150],[0,1,70]])

desloc = cv2.warpAffine(img,M,(cols,lin))
cv2.imshow('deslocamento',desloc)
'''

''''
#Rotacao

center = (cols/2,lin/2)
angle = -90
M = cv2.getRotationMatrix2D(center,angle,1)
rotate = cv2.warpAffine(img,M,(cols,lin))
cv2.imshow('rotacao',rotate)
'''


#aplicando um threshold

threshold_valor = 200
(T_valor,binary_threshold) = cv2.threshold(img,threshold_valor,255,cv2.THRESH_BINARY)
cv2.imshow('transformacao binaria',binary_threshold )

cv2.waitKey(0)
cv2.destroyAllWindows()