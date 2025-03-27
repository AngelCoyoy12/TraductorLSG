# config.py - Configuraciones importantes
import os
# Tamaño de las imágenes (ancho, alto)
TAMANO_IMAGEN = (224, 224)

# Obtiene la ruta donde está este archivo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Porcentaje de fotos para prueba (20%)
PORCENTAJE_PRUEBA = 0.2

# Rutas importantes
RUTA_IMAGENES = r"C:\Users\furio\OneDrive\Desktop\TRADUCTORLENGUAJESENAS\logica\imagenes_entrenamiento"
RUTA_MODELO = os.path.join(BASE_DIR, "..", "modelo_entrenado", "modelo.h5")
RUTA_ETIQUETAS = os.path.join(BASE_DIR, "..", "modelo_entrenado", "etiquetas.pkl")

# Configuración para el entrenamiento
EPOCAS = 10          # Veces que verá todas las fotos
LOTE = 32            # Cantidad de fotos que analiza a la vez