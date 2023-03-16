import cv2

classificadorFace = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
imagem = cv2.imread("imagens/rosto.png")

# cv2.imshow('rosto', imagem)
# cv2.waitKey()

imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
facesDetectadas = classificadorFace.detectMultiScale(imagemCinza, scaleFactor=1.2)

print(facesDetectadas)

for (x, y, l, a) in facesDetectadas:
    imagem = cv2.rectangle(imagem, (x, y), (x + l , y + a), (0,0,255), 2)

cv2.imshow('rosto', imagem)
cv2.waitKey()