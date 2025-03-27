import cv2
import numpy as np
import tensorflow as tf
import pickle
import os
import mediapipe as mp
from config import RUTA_MODELO, RUTA_ETIQUETAS, TAMANO_IMAGEN


class TraductorSenas:
    def __init__(self):
        try:
            # Cargar modelo y etiquetas
            self.modelo = tf.keras.models.load_model(RUTA_MODELO)
            with open(RUTA_ETIQUETAS, 'rb') as f:
                self.letras = pickle.load(f)

            # Inicializar MediaPipe Hands
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7
            )
            self.mp_drawing = mp.solutions.drawing_utils

            # Inicializar cámara
            self.camara = cv2.VideoCapture(0)
            if not self.camara.isOpened():
                raise RuntimeError("No se pudo abrir la cámara")

            print("Traductor listo. Presiona 'q' para salir.")

        except Exception as e:
            print(f"\nERROR durante inicialización: {str(e)}")
            self.cerrar_recursos()
            raise

    def procesar_fotograma(self, frame):
        try:
            # Detectar manos
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            # Dibujar landmarks si se detecta mano
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    # Obtener coordenadas para el cuadro delimitador
                    h, w, _ = frame.shape
                    x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                    y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                    x_min, x_max = int(min(x_coords)), int(max(x_coords))
                    y_min, y_max = int(min(y_coords)), int(max(y_coords))

                    # Dibujar cuadro delimitador (ampliado)
                    padding = 20
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(w, x_max + padding)
                    y_max = min(h, y_max + padding)

                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    # Recortar región de la mano y predecir
                    hand_roi = frame[y_min:y_max, x_min:x_max]
                    if hand_roi.size > 0:
                        img = cv2.resize(hand_roi, TAMANO_IMAGEN)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
                        img = np.expand_dims(img, axis=0)

                        prediccion = self.modelo.predict(img, verbose=0)
                        letra = self.letras[np.argmax(prediccion)]
                        confianza = np.max(prediccion)

                        # Mostrar letra cerca del cuadro
                        cv2.putText(frame, letra, (x_min, y_min - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        return letra

            return None

        except Exception as e:
            print(f"Error durante procesamiento: {str(e)}")
            return None

    def ejecutar(self):
        try:
            while True:
                ret, frame = self.camara.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                _ = self.procesar_fotograma(frame)

                cv2.imshow("Traductor", frame)

                if cv2.waitKey(1) in [ord('q'), 27]:
                    break

        finally:
            self.cerrar_recursos()

    def cerrar_recursos(self):
        if hasattr(self, 'camara') and self.camara.isOpened():
            self.camara.release()
        if hasattr(self, 'hands'):
            self.hands.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    traductor = TraductorSenas()
    traductor.ejecutar()