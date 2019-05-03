import cv2
import numpy

cap = cv2.VideoCapture('Vozão do meu coração.avi')

while (cap.isOpened()):
    ret,frame = cap.read()
    cv2.imshow('vid',frame)
    if cv2.waitKey(1) & 0xFF == ('q'):
        break
cap.release()
cv2.destroyAllWindows()
