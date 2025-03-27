# preparar_imagenes.py
import os
import cv2
from config import RUTA_IMAGENES, TAMANO_IMAGEN


def crear_carpetas():
    """Crea las carpetas necesarias si no existen"""
    if not os.path.exists(RUTA_IMAGENES):
        os.makedirs(RUTA_IMAGENES)
        print(f"Carpeta creada: {RUTA_IMAGENES}")

    # Ejemplo: crea carpetas para A, B, C si no existen
    for letra in ['A', 'B', 'C']:
        carpeta_letra = os.path.join(RUTA_IMAGENES, letra)
        if not os.path.exists(carpeta_letra):
            os.makedirs(carpeta_letra)
            print(f"Carpeta creada: {carpeta_letra}")


def procesar_imagenes():
    """Procesa todas las imágenes en la carpeta de entrenamiento"""
    crear_carpetas()  # Asegura que las carpetas existan

    print("\nPreparando imágenes...")
    print(f"Buscando en: {RUTA_IMAGENES}")

    for letra in os.listdir(RUTA_IMAGENES):
        letra_path = os.path.join(RUTA_IMAGENES, letra)

        if os.path.isdir(letra_path):
            print(f"\nProcesando letra: {letra}")
            print(f"En carpeta: {letra_path}")

            for img_name in os.listdir(letra_path):
                img_path = os.path.join(letra_path, img_name)

                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"¡Error! No se pudo leer: {img_path}")
                        continue

                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, TAMANO_IMAGEN)
                    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    print(f"✓ {img_name} procesada")

                except Exception as e:
                    print(f"✗ Error con {img_name}: {str(e)}")

    print("\n¡Preparación completada!")


if __name__ == "__main__":
    procesar_imagenes()
    input("Presiona Enter para salir...")  # Para que no se cierre la ventana
