import numpy as np
import  cv2

img = cv2.imread('ceara.png')



'''
#Gaussiano
matrix = (7,7)
blur = cv2.GaussianBlur(img, matrix,0)
cv2.imshow('blurred',blur)
'''



'''
#Mediana remover ruídos  e imperfeições
kern = 3

median =  cv2.medianBlur(img,kern)

cv2.imshow('transf mediana',median)
'''

'''
#filtro bilateral
dimpixel =  7  #dimensao do pixel central
color = 100
space = 100

filter  = cv2.bilateralFilter(img, dimpixel,color,space)

cv2.imshow('filtro bilateral',filter)
'''

#Edge dection

thresholdval1 = 50
thresholdval2 = 100

canny = cv2.Canny(img,thresholdval1,thresholdval2)
cv2.imshow('canny',canny)
cv2.waitKey(0)
cv2.destroyAllWindows()