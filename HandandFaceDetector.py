import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone import FaceDetectionModule

webcam = cv2.VideoCapture(0)

rastreador = HandDetector(detectionCon=0.8, maxHands=2)
face_detector = FaceDetectionModule.FaceDetector()


while True:
    exito, imagen = webcam.read()
    coordenadas, imagen_manos = rastreador.findHands(imagen)
    imagen_manos, list_faces = face_detector.findFaces(imagen_manos)

    # Mostrar coordenadas
    print(coordenadas)
 
    cv2.imshow("Proyecto 4 - IA", imagen_manos)

    if cv2.waitKey(1) != -1:
        break

webcam.release()
cv2.destroyAllWindows()
