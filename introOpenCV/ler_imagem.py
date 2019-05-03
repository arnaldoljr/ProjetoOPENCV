import numpy as np
import cv2

img = cv2.imread('ceara.png')
img =  cv2.imwrite('imagemcop.jpg',img)
cv2.imshow('original', img)
cv2.waitKey(0)
cv2.destroyAllWindows()