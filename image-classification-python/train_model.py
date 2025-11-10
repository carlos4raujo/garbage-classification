#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

# Configuración inicial
IMAGE_SIZE = 100  # Tamaño al que se redimensionarán todas las imágenes

# Obtener las clases de basura desde los directorios
data_dir = "data"
clases = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
print("Clases de basura encontradas:", clases)

# Cargar el dataset desde la carpeta 'data'
# Configuramos el dataset para cargar imágenes en escala de grises y mezcladas
train_ds = tf.keras.utils.image_dataset_from_directory(
    "data",                               # Carpeta raíz con subcarpetas por clase
    image_size=(IMAGE_SIZE, IMAGE_SIZE),  # Redimensionar todas las imágenes
    batch_size=1,                         # Tamaño del lote
    color_mode='grayscale',               # Convertir a escala de grises
    shuffle=True                          # Mezclar las imágenes aleatoriamente
)

# Guardar los nombres de las clases para usar en predicciones
class_names = train_ds.class_names

# Desempaquetar el dataset en elementos individuales
train_ds = train_ds.unbatch()

# Visualizar algunas imágenes del dataset
plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(train_ds.take(9)):
    ax = plt.subplot(3, 3, i + 1)
    # Mostrar la imagen en escala de grises
    plt.imshow(image.numpy().astype("uint8").squeeze(), cmap='gray')
    plt.title(class_names[label.numpy()])
    plt.axis("off")
plt.show()

# Preparar los datos para el entrenamiento
data_train = []
for image, label in train_ds:
    # Convertir la imagen a NumPy y asegurar formato correcto
    image_np = image.numpy().astype(np.uint8)
    # Asegurar que la imagen tiene la forma (altura, ancho, 1)
    if image_np.ndim == 2:
        image_np = np.expand_dims(image_np, axis=-1)
    data_train.append([image_np, label])

print("Total de imágenes cargadas:", len(data_train))

# Separar imágenes y etiquetas en listas diferentes
images_list = []  # Lista para almacenar las imágenes
labels_list = []  # Lista para almacenar las etiquetas

for image, label in data_train:
    images_list.append(image)
    labels_list.append(label)

# Normalizar las imágenes (convertir a valores entre 0 y 1)
images_list = np.array(images_list).astype(float) / 255
labels_list = np.array(labels_list)

# Visualizar imágenes sin aumento de datos
plt.figure(figsize=(20, 8))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(images_list[i].reshape(IMAGE_SIZE, IMAGE_SIZE), cmap="gray")
plt.show()

# Configurar el generador de aumento de datos
# Esto ayuda a prevenir el sobreajuste y mejorar la generalización
datagen = ImageDataGenerator(
    rotation_range=30,            # Rotación aleatoria hasta 30 grados
    width_shift_range=0.2,        # Desplazamiento horizontal
    height_shift_range=0.2,       # Desplazamiento vertical
    shear_range=15,              # Transformación de cizalladura
    zoom_range=[0.7, 1.4],       # Rango de zoom
    horizontal_flip=True,        # Volteo horizontal
    vertical_flip=True,          # Volteo vertical
    fill_mode='nearest'          # Método para rellenar píxeles nuevos
)

# Ajustar el generador a nuestros datos
datagen.fit(images_list)

# Visualizar ejemplos de imágenes aumentadas
plt.figure(figsize=(20,8))
for image, label in datagen.flow(images_list, labels_list, batch_size=10, shuffle=False):
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(image[i].reshape(IMAGE_SIZE, IMAGE_SIZE), cmap="gray")
    break
plt.show()

# Dividir los datos en conjuntos de entrenamiento y validación (85% - 15%)
train_size = int(len(images_list) * 0.85)
images_train = images_list[:train_size]
images_validation = images_list[train_size:]
labels_train = labels_list[:train_size]
labels_validation = labels_list[train_size:]

# Crear el generador de datos de entrenamiento
data_gen_train = datagen.flow(images_train, labels_train, batch_size=32)

# Definir la arquitectura del modelo CNN
model = tf.keras.models.Sequential([
    # Capa de entrada
    tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
    
    # Primera capa convolucional
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    # Segunda capa convolucional
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    # Tercera capa convolucional
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    # Aplanar los datos para las capas densas
    tf.keras.layers.Flatten(),
    
    # Dropout para reducir el sobreajuste
    tf.keras.layers.Dropout(0.3),
    
    # Capas densas finales
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')  # 5 clases de salida
])

# Compilar el modelo
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenar el modelo
history = model.fit(
    data_gen_train,
    epochs=150,
    batch_size=32,
    validation_data=(images_validation, labels_validation),
    verbose=1
)

# Evaluar el modelo con el conjunto de validación
loss, accuracy = model.evaluate(images_validation, labels_validation, verbose=1)
print(f"Pérdida en validación: {loss:.4f}")
print(f"Precisión en validación: {accuracy:.4f}")

# Función para probar el modelo con una imagen
def predict_image(img_path):
    # Cargar y preprocesar la imagen
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=(0, -1))  # Añadir dimensiones de lote y canal
    
    # Realizar la predicción
    pred = model.predict(img)
    label_pred = np.argmax(pred)
    
    print("Clase predicha:", class_names[label_pred])
    return class_names[label_pred]

# Probar el modelo con una imagen de prueba
test_image = "test_images/image-1.jpg"
predicted_class = predict_image(test_image)

# Guardar el modelo entrenado
model.save('models/image-classification.h5')
print("Modelo guardado exitosamente en 'models/image-classification.h5'")