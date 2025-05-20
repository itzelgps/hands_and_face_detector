import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone import FaceDetectionModule

webcam = cv2.VideoCapture(0)

rastreador = HandDetector(detectionCon=0.8, maxHands=2)
face_detector = FaceDetectionModule.FaceDetector()


while True:
    exito, imagen = webcam.read()

    if not exito:
        continue
    
    # Detectar manos
    coordenadas, imagen_manos = rastreador.findHands(imagen)

    # Detectar cara
    imagen_manos, list_faces = face_detector.findFaces(imagen_manos)

    if coordenadas:
        for mano in coordenadas:
            dedos_arriba = rastreador.fingersUp(mano)
            print("Dedos arriba:", dedos_arriba)

            # Si el índice esta levantado, ejecutar una acción
            if dedos_arriba == [0, 1, 0, 0, 0]: # Solo el indice levantado
                print("¡Gesto detectado: Selección!")
            elif dedos_arriba == [1, 1, 0, 0, 0]: # Pulgar e índice arriba
                print("¡Gesto detectado: Zoom!") 
            elif dedos_arriba == [1, 1, 1, 1, 1]: # Mano abierta
                print("¡Gesto detectado: Pausa!")

    # Mostrar coordenadas
    #print(coordenadas)
 
    cv2.imshow("Proyecto 4 - IA", imagen_manos)

    if cv2.waitKey(1) != -1:
        break

webcam.release()
cv2.destroyAllWindows()
