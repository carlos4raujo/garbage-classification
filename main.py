# %%
import os

data_dir = "data" 
clases = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
print(clases)


# %%
import tensorflow as tf
import numpy as np

IMAGE_SIZE = 100

# Cargar el dataset desde la carpeta 'data'
train_ds = tf.keras.utils.image_dataset_from_directory(
    "data",                     # Carpeta raíz con subcarpetas por clase
    image_size=(IMAGE_SIZE, IMAGE_SIZE),      # Redimensionar todas las imágenes
    batch_size=1,              # Tamaño del lote
    color_mode='grayscale',     # Convertir a escala de grises
    shuffle=True                # Mezclar las imágenes
)

class_names = train_ds.class_names  # guárdalo aquí

train_ds = train_ds.unbatch()


# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))

for i, (image, label) in enumerate(train_ds.take(9)):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image.numpy().astype("uint8").squeeze(), cmap='gray')  # quitar canal extra y usar escala de grises
    plt.title(class_names[label.numpy()])
    plt.axis("off")

plt.show()


# %%
data_train = []

for image, label in train_ds:
    # Convertimos la imagen a NumPy con tipo uint8
    image_np = image.numpy().astype(np.uint8)
    # Aseguramos la forma (H, W, 1)
    if image_np.ndim == 2:
        image_np = np.expand_dims(image_np, axis=-1)
    
    # Guardamos como [imagen, label]
    data_train.append([image_np, label])


# %%
len(data_train)

# %%
images_list = [] #Imagenes de entrada
labels_list = [] #Etiquetas

for image, label in data_train:
    images_list.append(image)
    labels_list.append(label)

# %%
images_list = np.array(images_list).astype(float) / 255

# %%
labels_list = np.array(labels_list)

# %%
#ver las imagenes de la variable X sin modificaciones por aumento de datos
plt.figure(figsize=(20, 8))

for i in range(10):
  plt.subplot(2, 5, i+1)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(images_list[i].reshape(IMAGE_SIZE, IMAGE_SIZE), cmap="gray")

# %%
#Realizar el aumento de datos con varias transformaciones. Al final, graficar 10 como ejemplo
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=15,
    zoom_range=[0.7, 1.4],
    horizontal_flip=True,
    vertical_flip=True,
    # brightness_range=[0.8, 1.2],
    fill_mode='nearest',
)

datagen.fit(images_list)

plt.figure(figsize=(20,8))

for image, label in datagen.flow(images_list, labels_list, batch_size=10, shuffle=False):
  for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image[i].reshape(IMAGE_SIZE, IMAGE_SIZE), cmap="gray")
  break

# %%
len(images_list) - (len(images_list) * .85) #19700

# %%
len(images_list) * .85 #2147
len(images_list) - 2147 #379

images_train = images_list[:2147]
images_validation = images_list[2147:]

labels_train = labels_list[:2147]
labels_validation = labels_list[2147:]

# %%
data_gen_train = datagen.flow(images_train, labels_train, batch_size=32)

# %%
from tensorflow.keras.callbacks import TensorBoard

# %%
#Cargar la extension de tensorboard de colab
%load_ext tensorboard

# %%
tensorboard = TensorBoard(log_dir='logs/one')

# %%
#Ejecutar tensorboard e indicarle que lea la carpeta "logs"
%tensorboard --logdir logs

# %%
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),

  tf.keras.layers.Flatten(),
  tf.keras.layers.Dropout(0.3),
  tf.keras.layers.Dense(100, activation='relu'),
  tf.keras.layers.Dense(6, activation='softmax') 
])

# %%
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # o categorical si tienes one-hot
    metrics=['accuracy']
)


# %%
#Usar la funcion flow del generador para crear un iterador que podamos enviar como entrenamiento a la funcion FIT del modelo
model.fit(
    data_gen_train,
    epochs=150, batch_size=32,
    validation_data=(images_validation, labels_validation),
    steps_per_epoch=len(images_train) // 32,
    verbose=1,
    callbacks=[tensorboard]
)

# %%
loss, accuracy = model.evaluate(images_validation, labels_validation, verbose=1)
print(f"Pérdida: {loss:.4f}")
print(f"Precisión: {accuracy:.4f}")


# %%
import cv2

# Cargar y preprocesar la imagen
img_path = "test_images/image-1.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
img = img / 255.0
img = np.expand_dims(img, axis=(0, -1))  # (1, H, W, 1)

# Predecir
pred = model.predict(img)
label_pred = np.argmax(pred)

print("Clase predicha:", class_names[label_pred])


# %%
model.save('garbage-classification-cnn-ad.h5')

# %%
%pip install tensorflowjs

# %%
%mkdir output_folder

# %%
import numpy as np

np.object = np.object_

import tensorflowjs as tfjs

tfjs_target_dir = 'output_folder'

tfjs.converters.save_keras_model(model, tfjs_target_dir)


