import cv2
import numpy

cap = cv2.VideoCapture('Vozão do meu coração.avi')
<<<<<<< HEAD
#criar uma copia em outro formato
fourcc =  cv2.VideoWriter_fourcc(*'XVID')
fps = 30
framesize =  (720,480)
out = cv2.VideoWriter('exemplo.mp4',fourcc,fps,framesize)

=======
>>>>>>> efe4a48975327255d68fb6a27a39d80e244bd859

while (cap.isOpened()):
    ret,frame = cap.read()
    cv2.imshow('vid',frame)
    if cv2.waitKey(1) & 0xFF == ('q'):
        break
cap.release()
<<<<<<< HEAD
cv2.destroyAllWindows()
=======
cv2.destroyAllWindows()
>>>>>>> efe4a48975327255d68fb6a27a39d80e244bd859
