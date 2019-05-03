import numpy as np
import cv2

pic = np.zeros((500,500,3),dtype='uint8')

font  =  cv2.FONT_HERSHEY_DUPLEX

cv2.putText(pic,'UDEMY',(100,100),font,3, (255,255,255),4,cv2.LINE_8)

color =  (255,0,255)
cv2.circle(pic,(250,250),100,color)
cv2.line(pic,(350,350), (500,350),(0,0,255))
cv2.rectangle(pic,(0,0),(500,150),(123,200,98),3,lineType=8,shift=0)

cv2.imshow('dark', pic)
cv2.waitKey(0)
cv2.destroyAllWindows()