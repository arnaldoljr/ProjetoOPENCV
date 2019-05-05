import cv2
import numpy

cap = cv2.VideoCapture('Vozão do meu coração.avi')
#criar uma copia em outro formato
fourcc =  cv2.VideoWriter_fourcc(*'XVID')
fps = 30
framesize =  (720,480)
out = cv2.VideoWriter('exemplo.mp4',fourcc,fps,framesize)


while (cap.isOpened()):
    ret,frame = cap.read()
    cv2.imshow('vid',frame)
    if cv2.waitKey(1) & 0xFF == ('q'):
        break
cap.release()
cv2.destroyAllWindows()