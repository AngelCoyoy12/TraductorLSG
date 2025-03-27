import cv2
import numpy as np
import tensorflow as tf
import pickle
import os
from config import RUTA_MODELO, RUTA_ETIQUETAS, TAMANO_IMAGEN


class TraductorSenas:
    def __init__(self):
        try:
            # Cargar modelo y etiquetas con verificación
            if not os.path.exists(RUTA_MODELO):
                raise FileNotFoundError(f"No se encontró el modelo en {RUTA_MODELO}")
            if not os.path.exists(RUTA_ETIQUETAS):
                raise FileNotFoundError(f"No se encontraron etiquetas en {RUTA_ETIQUETAS}")

            self.modelo = tf.keras.models.load_model(RUTA_MODELO)
            with open(RUTA_ETIQUETAS, 'rb') as f:
                self.letras = pickle.load(f)

            # Iniciar cámara con reintentos
            self.camara = None
            for _ in range(3):  # 3 intentos
                self.camara = cv2.VideoCapture(0)
                if self.camara.isOpened():
                    break
                print("Reintentando abrir cámara...")

            if not self.camara or not self.camara.isOpened():
                raise RuntimeError("No se pudo abrir la cámara después de 3 intentos")

            print("Traductor listo. Presiona 'q' para salir.")

        except Exception as e:
            print(f"\nERROR durante inicialización: {str(e)}")
            self.cerrar_recursos()
            raise

    def procesar_fotograma(self, frame):
        """Procesa un fotograma para detectar la letra"""
        try:
            # Redimensionar y normalizar
            img = cv2.resize(frame, TAMANO_IMAGEN)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            img = np.expand_dims(img, axis=0)  # Añadir dimensión de batch

            # Predecir
            prediccion = self.modelo.predict(img, verbose=0)
            indice = np.argmax(prediccion)
            letra = self.letras[indice]
            confianza = prediccion[0][indice]

            return letra, float(confianza)

        except Exception as e:
            print(f"Error durante predicción: {str(e)}")
            return None, 0.0

    def ejecutar(self):
        """Bucle principal del traductor"""
        try:
            while True:
                try:
                    # Capturar fotograma
                    ret, frame = self.camara.read()
                    if not ret:
                        print("Error al capturar imagen - reintentando...")
                        continue

                    # Voltear horizontalmente (espejo)
                    frame = cv2.flip(frame, 1)

                    # Procesar y mostrar resultados
                    letra, confianza = self.procesar_fotograma(frame)

                    # Mostrar resultados
                    if letra:
                        cv2.putText(frame, f"Letra: {letra}", (20, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f"Confianza: {confianza:.2f}", (20, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Mostrar ventana
                    cv2.imshow("Traductor de Señas", frame)

                    # Salir con 'q' o ESC
                    key = cv2.waitKey(1)
                    if key == ord('q') or key == 27:
                        break

                except KeyboardInterrupt:
                    print("\nInterrupción por usuario detectada")
                    break
                except Exception as e:
                    print(f"Error durante ejecución: {str(e)}")
                    continue

        finally:
            self.cerrar_recursos()

    def cerrar_recursos(self):
        """Libera todos los recursos correctamente"""
        if hasattr(self, 'camara') and self.camara and self.camara.isOpened():
            self.camara.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        traductor = TraductorSenas()
        traductor.ejecutar()
    except Exception as e:
        print(f"\nError fatal: {str(e)}")
    finally:
        print("\nPrograma terminado correctamente")