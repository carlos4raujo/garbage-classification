import * as tf from '@tensorflow/tfjs-node';
import fs from 'fs';

// === 1. Cargar dataset ===
const rawData = JSON.parse(fs.readFileSync('./dataset.json', 'utf8'));

const texts = [];
const labels = [];
const labelNames = Object.keys(rawData);

for (const [label, examples] of Object.entries(rawData)) {
  for (const ex of examples) {
    texts.push(ex.replace(/_/g, ' '));
    labels.push(label);
  }
}

// === 2. Crear tokenizador simple ===
const vocab = new Set();
texts.forEach(t => t.split(/\s+/).forEach(w => vocab.add(w)));
const wordIndex = {};
Array.from(vocab).forEach((w, i) => (wordIndex[w] = i + 1));

const sequences = texts.map(t =>
  t.split(/\s+/).map(w => wordIndex[w] || 0)
);

const maxLen = Math.max(...sequences.map(seq => seq.length));
const padded = sequences.map(seq => {
  const arr = new Array(maxLen).fill(0);
  seq.forEach((n, i) => (arr[i] = n));
  return arr;
});

// === 3. Convertir a tensores ===
const X = tf.tensor2d(padded);
const yIdx = labels.map(l => labelNames.indexOf(l));
const y = tf.oneHot(tf.tensor1d(yIdx, 'int32'), labelNames.length);

// === 4. Crear modelo ===
const model = tf.sequential();
model.add(tf.layers.embedding({
  inputDim: vocab.size + 1,
  outputDim: 16,
  inputLength: maxLen
}));
model.add(tf.layers.globalAveragePooling1d());
model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
model.add(tf.layers.dense({ units: labelNames.length, activation: 'softmax' }));

model.compile({
  loss: 'categoricalCrossentropy',
  optimizer: 'adam',
  metrics: ['accuracy']
});

// === 5. Entrenar ===
await model.fit(X, y, {
  epochs: 200,
  verbose: 1
});

// === 6. Guardar modelo y tokenizer ===
await model.save('file://./model');

fs.writeFileSync('./word_index.json', JSON.stringify(wordIndex, null, 2));
fs.writeFileSync('./labels.json', JSON.stringify(labelNames, null, 2));

console.log('✅ Modelo entrenado y guardado en ./model');
