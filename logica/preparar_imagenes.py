import os
import cv2
from config import RUTA_IMAGENES, TAMANO_IMAGEN


def crear_carpetas():
    if not os.path.exists(RUTA_IMAGENES):
        os.makedirs(RUTA_IMAGENES)
        print(f"Carpeta creada: {RUTA_IMAGENES}")

    for letra in ['A', 'B', 'C']:
        carpeta_letra = os.path.join(RUTA_IMAGENES, letra)
        if not os.path.exists(carpeta_letra):
            os.makedirs(carpeta_letra)
            print(f"Carpeta creada: {carpeta_letra}")


def procesar_imagenes():
    crear_carpetas()
    print(f"Preparando imágenes en: {RUTA_IMAGENES}")

    procesadas = 0
    errores = 0

    for letra in os.listdir(RUTA_IMAGENES):
        letra_path = os.path.join(RUTA_IMAGENES, letra)

        if os.path.isdir(letra_path):
            for img_name in os.listdir(letra_path):
                img_path = os.path.join(letra_path, img_name)

                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        errores += 1
                        continue

                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, TAMANO_IMAGEN)
                    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    procesadas += 1

                except Exception:
                    errores += 1

    print(f"✓ Imágenes procesadas: {procesadas}")
    print(f"✗ Errores: {errores}")
    print("Preparación completada")


if __name__ == "__main__":
    procesar_imagenes()
    input("Presiona Enter para salir...")

