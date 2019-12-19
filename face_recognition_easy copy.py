import face_recognition
import cv2
import numpy as np


#extraer imagenes de un video
video_capture = cv2.VideoCapture('Will.mp4')

#extraer imagenes de la webcam
#video_capture = cv2.VideoCapture(0)

if (video_capture.isOpened()== False): 
  print("Error opening video stream or file")

# Carga la imagen de entrenamiento
will_smith_image = face_recognition.load_image_file("img/will_smith.jpg")
# Codifica la cara de entrenamiento 
will_smith_face_encoding = face_recognition.face_encodings(will_smith_image)[0]


# Crea un arreglo con las distintas caras codificadas
known_face_encodings = [
    will_smith_face_encoding
]
# Crea un arreglo de los nombres en orden de las caras codificadas
known_face_names = [
    "Will Smith",
]

# Inicializa algunas variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Toma los fotogramas del capturador
    ret, frame = video_capture.read()

    # Redimensiona el fotograma a 1/4 de resolucion para mejor rendimiento
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convierte la imagen de BGR color a RGB 
    rgb_small_frame = small_frame[:, :, ::-1]

    # Solamente procesa fotogramas intercalando entre uno y otro
    if process_this_frame:
        # encuentra y codifica todas las caras existentes en el fotograma
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # compara las caras codificadas con las caras entrenadas
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # usa la distancia entre las caras para encontrar la mayor probabilidad de match
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Muestra los resultados
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Escala nuevamente el fotograma a la resolucion original
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Dibuja un rectangulo al rededor del rostro
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Dibuja el nombre del rostro encontrado
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Muestra el resultado de la imagen
    cv2.imshow('Video', frame)

    # Presiona 'Q' para terminar el proceso
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera el video o la webcam
video_capture.release()
cv2.destroyAllWindows()
