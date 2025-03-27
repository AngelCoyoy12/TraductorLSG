import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle
from config import *
import sys


def aumentar_datos(imagenes, etiquetas, factor=20):
    datagen = ImageDataGenerator(
        rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
        zoom_range=0.2, horizontal_flip=True, brightness_range=[0.8, 1.2]
    )

    aumentadas = []
    etiquetas_aumentadas = []

    for img, label in zip(imagenes, etiquetas):
        img = img.reshape((1,) + img.shape)
        i = 0
        for batch in datagen.flow(img, batch_size=1):
            aumentadas.append(batch[0])
            etiquetas_aumentadas.append(label)
            i += 1
            if i >= factor:
                break

    return np.array(aumentadas), np.array(etiquetas_aumentadas)


def cargar_imagenes():
    imagenes, etiquetas, letras = [], [], []

    if not os.path.exists(RUTA_IMAGENES):
        print(f"Error: Carpeta {RUTA_IMAGENES} no existe.")
        sys.exit(1)

    letras_disponibles = sorted([d for d in os.listdir(RUTA_IMAGENES)
                                 if os.path.isdir(os.path.join(RUTA_IMAGENES, d))])

    if not letras_disponibles:
        print("Error: No hay subcarpetas de letras.")
        sys.exit(1)

    print(f"Letras detectadas: {', '.join(letras_disponibles)}")

    for letra in letras_disponibles:
        letra_path = os.path.join(RUTA_IMAGENES, letra)
        imagenes_letra = [f for f in os.listdir(letra_path)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for img_name in imagenes_letra:
            img_path = os.path.join(letra_path, img_name)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, TAMANO_IMAGEN)
                img = img / 255.0

                if letra not in letras:
                    letras.append(letra)
                etiqueta = letras.index(letra)

                imagenes.append(img)
                etiquetas.append(etiqueta)

            except Exception as e:
                print(f"Error procesando {img_name}: {str(e)}")

    if not imagenes:
        print("Error: No se cargaron imágenes válidas.")
        sys.exit(1)

    return np.array(imagenes), np.array(etiquetas), letras


def crear_modelo(num_clases):
    modelo = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(TAMANO_IMAGEN[0], TAMANO_IMAGEN[1], 3)),
        MaxPooling2D(2, 2),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_clases, activation='softmax')
    ])

    modelo.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return modelo


def main():
    try:
        # Cargar imágenes
        X, y, letras = cargar_imagenes()
        print(f"Imágenes cargadas: {len(X)} | Letras: {letras}")

        # Aumentar datos si es necesario
        if len(X) < 50:
            X_aug, y_aug = aumentar_datos(X, y, factor=20)
            X, y = np.concatenate((X, X_aug)), np.concatenate((y, y_aug))
            print(f"Total después de aumento: {len(X)} imágenes")

        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=PORCENTAJE_PRUEBA, random_state=42, stratify=y
        )

        # Crear y entrenar modelo
        modelo = crear_modelo(len(letras))
        historia = modelo.fit(
            X_train, y_train,
            epochs=EPOCAS,
            batch_size=LOTE,
            validation_data=(X_test, y_test),
            verbose=1
        )

        # Guardar modelo y etiquetas
        os.makedirs(os.path.dirname(RUTA_MODELO), exist_ok=True)
        modelo.save(RUTA_MODELO)

        with open(RUTA_ETIQUETAS, 'wb') as f:
            pickle.dump(letras, f)

        # Mostrar precisión final
        loss, accuracy = modelo.evaluate(X_test, y_test, verbose=0)
        print(f"✓ Modelo entrenado | Precisión: {accuracy:.2%}")
        print(f"Modelo guardado en: {RUTA_MODELO}")

    except Exception as e:
        print(f"Error durante entrenamiento: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
    input("Presiona Enter para salir...")