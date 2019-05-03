import cv2

imagem1 = cv2.imread('Imagens teste canecas/teste04.jpg')
imagem2 = cv2.imread('Imagens teste canecas/teste04.jpg')
imagem3 = cv2.imread('Imagens teste canecas/teste04.jpg')

classificador1 = cv2.CascadeClassifier('/Users/arnaldoljr/PycharmProjects/ProjDect/cascade-caneca1.xml')
classificador2 = cv2.CascadeClassifier('/Users/arnaldoljr/PycharmProjects/ProjDect/cascade-caneca2.xml')
classificador3 = cv2.CascadeClassifier('/Users/arnaldoljr/PycharmProjects/ProjDect/cascade-caneca3.xml')

imagemcinza1 = cv2.cvtColor(imagem1, cv2.COLOR_BGR2GRAY)
imagemcinza2 = cv2.cvtColor(imagem2, cv2.COLOR_BGR2GRAY)
imagemcinza3 = cv2.cvtColor(imagem3, cv2.COLOR_BGR2GRAY)

deteccoes1 = classificador1.detectMultiScale(imagemcinza1, scaleFactor=1.2, minNeighbors=4)
deteccoes2 = classificador2.detectMultiScale(imagemcinza2, scaleFactor=1.2, minNeighbors=4)
deteccoes3 = classificador3.detectMultiScale(imagemcinza3, scaleFactor=1.2, minNeighbors=4)

for (x, y, l, a) in deteccoes1:
    cv2.rectangle(imagem1, (x, y), (x + l, y + a), (0,255,0), 2)

for (x, y, l, a) in deteccoes2:
    cv2.rectangle(imagem2, (x, y), (x + l, y + a), (0,255,0), 2)

for (x, y, l, a) in deteccoes3:
    cv2.rectangle(imagem3, (x, y), (x + l, y + a), (0,255,0), 2)

cv2.imshow('Classificador 1', imagem1)
cv2.imshow('Classificador 2', imagem2)
cv2.imshow('Classificador 3', imagem3)
cv2.waitKey(0)
cv2.destroyAllWindows()