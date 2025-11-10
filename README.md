# Clasificación de Basura (Garbage Classification)

Este proyecto implementa un sistema de clasificación de basura utilizando técnicas de aprendizaje profundo para clasificar tanto imágenes como texto descriptivo. El proyecto está dividido en tres componentes principales:

## 📱 Aplicación Web (web/)

Esta parte del proyecto proporciona una interfaz de usuario interactiva construida con React donde puedes probar los modelos de clasificación.

### Tecnologías Utilizadas
- React
- Vite
- TensorFlow.js
- CSS para estilos

### Estructura
```
web/
├── src/
│   ├── components/        # Componentes React
│   ├── constants/        # Constantes y configuraciones
│   ├── utils/           # Funciones utilitarias
│   └── assets/         # Recursos estáticos
└── public/
    └── assets/
        └── models/     # Modelos entrenados
```

### Cómo Ejecutar
1. Navega al directorio web:
```bash
cd web
```

2. Instala las dependencias:
```bash
npm install
```

3. Inicia el servidor de desarrollo:
```bash
npm run dev
```

4. Abre tu navegador en `http://localhost:5173`

## 📝 Clasificación de Texto (text-classification-node/)

Este componente se encarga de entrenar un modelo de clasificación de texto utilizando TensorFlow.js en Node.js.

### Librerías Principales
- @tensorflow/tfjs-node: Para el entrenamiento del modelo
- fs: Para manejo de archivos

### Estructura
```
text-classification-node/
├── dataset.json         # Dataset de entrenamiento
├── train.js            # Script de entrenamiento
└── predict.js          # Script para predicciones
```

### Cómo Ejecutar
1. Navega al directorio:
```bash
cd text-classification-node
```

2. Instala las dependencias:
```bash
npm install
```

3. Entrena el modelo:
```bash
node train.js
```

4. Para hacer predicciones:
```bash
node predict.js
```

## 🖼️ Clasificación de Imágenes (image-classification-python/)

Este componente implementa un modelo de clasificación de imágenes utilizando TensorFlow y Keras en Python.

### Librerías Principales
- tensorflow: Framework principal de machine learning
- numpy: Para procesamiento numérico
- opencv-python: Para procesamiento de imágenes
- matplotlib: Para visualización de datos

### Entorno Virtual y Configuración
1. Crear y activar entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # En macOS/Linux
# o
venv\Scripts\activate     # En Windows
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

### Estructura
```
image-classification-python/
├── data/               # Imágenes de entrenamiento por categoría
├── test_images/       # Imágenes para pruebas
├── train_model.py     # Script de entrenamiento
└── train_model.ipynb  # Notebook interactivo
```

### Diferencia entre train_model.py y train_model.ipynb

- **train_model.py**: 
  - Script optimizado para entrenamiento en producción
  - Ejecución directa sin interfaz interactiva
  - Mejor para automatización y pipeline de entrenamiento
  - Comentarios detallados en español
  - Código más limpio y estructurado
  - Ideal para entrenamientos repetitivos

- **train_model.ipynb**:
  - Notebook interactivo para experimentación
  - Visualización de resultados intermedios
  - Ideal para prototipado y ajuste de parámetros
  - Permite ejecución por celdas para análisis paso a paso
  - Incluye gráficas y visualizaciones
  - Perfecto para exploración de datos y pruebas

### Cómo Ejecutar

1. Para entrenamiento directo:
```bash
python train_model.py
```

2. Para experimentación interactiva:
```bash
jupyter notebook train_model.ipynb
```

## 🌟 Características del Proyecto

- Clasificación dual: texto e imágenes
- Interfaz web interactiva
- Modelos pre-entrenados incluidos
- Soporte para múltiples categorías de basura
- Documentación detallada en español
- Fácil de desplegar y usar

## 📁 Categorías de Clasificación

- Cartón (Cardboard)
- Vidrio (Glass)
- Metal
- Papel (Paper)
- Plástico (Plastic)