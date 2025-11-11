import * as tf from '@tensorflow/tfjs-node';
import fs from 'fs';
import path from 'path';

// Config
const IMAGE_SIZE = 100;
const DATA_DIR = './data';

// Obtener clases (subcarpetas)
const clases = fs.readdirSync(DATA_DIR).filter(d => 
  fs.statSync(path.join(DATA_DIR, d)).isDirectory()
);
console.log('Clases encontradas:', clases);

// Cargar imágenes en tensores
async function loadImages() {
  const images = [];
  const labels = [];

  for (let i = 0; i < clases.length; i++) {
    const classDir = path.join(DATA_DIR, clases[i]);
    const files = fs.readdirSync(classDir);
    for (const file of files) {
      const imgPath = path.join(classDir, file);
      const imgBuffer = fs.readFileSync(imgPath);
      const imgTensor = tf.node.decodeImage(imgBuffer, 1) // 1 = grayscale
        .resizeNearestNeighbor([IMAGE_SIZE, IMAGE_SIZE])
        .toFloat()
        .div(255.0);
      images.push(imgTensor);
      labels.push(i);
    }
  }

  return {
    images: tf.stack(images),
    labels: tf.tensor1d(labels, 'int32'),
  };
}

const { images, labels } = await loadImages();

// Crear modelo CNN
const model = tf.sequential();
model.add(tf.layers.conv2d({
  inputShape: [IMAGE_SIZE, IMAGE_SIZE, 1],
  filters: 32, kernelSize: 3, activation: 'relu'
}));
model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
model.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu' }));
model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
model.add(tf.layers.conv2d({ filters: 128, kernelSize: 3, activation: 'relu' }));
model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
model.add(tf.layers.flatten());
model.add(tf.layers.dropout({ rate: 0.3 }));
model.add(tf.layers.dense({ units: 100, activation: 'relu' }));
model.add(tf.layers.dense({ units: clases.length, activation: 'softmax' }));

model.compile({
  optimizer: tf.train.adam(),
  loss: 'sparseCategoricalCrossentropy',
  metrics: ['accuracy']
});

// Entrenar
await model.fit(images, labels, {
  epochs: 30,
  batchSize: 32,
  validationSplit: 0.15,
  shuffle: true
});

// Guardar modelo para uso en navegador
await model.save('file://./models/image-classification');
console.log('✅ Modelo guardado');