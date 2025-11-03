# Garbage Classification (CNN)

This repository contains a simple TensorFlow-based convolutional neural network (CNN) for classifying images of garbage into several categories. The primary training script is `main.py` (originally developed in a Jupyter environment), and the trained Keras model is saved as `garbage-classification-cnn-ad.h5`.

## Project structure

Important files and folders:

- `main.py` — training / evaluation / export script (contains Jupyter magics; see Notes)
- `garbage-classification-cnn-ad.h5` — example saved Keras model
- `data/` — dataset root. Each class must be a subdirectory with images, e.g.:
  - `data/cardboard/`
  - `data/glass/`
  - `data/metal/`
  - `data/paper/`
  - `data/plastic/`
  - `data/trash/`
- `test_images/` — example images used for quick predictions (e.g. `test_images/image-1.jpg`)
- `output_folder/` — where the TensorFlow.js-converted model is saved by the script
- `logs/` — TensorBoard logs

## Quick overview of `main.py`

- Loads images from the `data/` directory using `tf.keras.utils.image_dataset_from_directory`.
- Converts images to grayscale and resizes to 100x100.
- Builds a small CNN, trains it with data augmentation (via `ImageDataGenerator`), and logs to TensorBoard.
- Evaluates on a validation split and demonstrates a single-image prediction.
- Saves the Keras model (`.h5`) and converts it to a TensorFlow.js model in `output_folder/`.

## Requirements

Create a virtual environment and install dependencies. A `requirements.txt` is provided.

Recommended (macOS, zsh):

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Running

Notes: `main.py` was written with interactive Jupyter usage in mind and includes IPython magic commands such as `%load_ext tensorboard`, `%tensorboard --logdir logs`, and `%pip install ...`. You have two options:

1) Run inside a Jupyter notebook (recommended as-is):

   - Open `main.ipynb` in Jupyter Lab / Notebook or convert `main.py` to a notebook.
   - Run the notebook cells. The `%tensorboard` magic will open a TensorBoard session inside the notebook.

2) Run as a plain Python script (remove or replace IPython magics):

   - Edit `main.py` and remove or comment out lines that start with `%` (magic commands). Replace `%pip install tensorflowjs` with `pip install tensorflowjs` done beforehand, and remove `%tensorboard` / `%load_ext` lines.
   - Run:

```bash
python main.py
```

If you prefer to only run prediction using an existing saved model (`garbage-classification-cnn-ad.h5`), ensure the model file exists in the repository root and run a small script that loads the model and runs inference on an image.

## TensorBoard

If running in a notebook, the existing `%tensorboard --logdir logs` magic works. If running outside a notebook, start TensorBoard from the project root:

```bash
tensorboard --logdir logs --port 6006
# then open http://localhost:6006
```

## TensorFlow.js export

The script saves a Keras `.h5` model and attempts to convert it to TensorFlow.js format under `output_folder/` using the `tensorflowjs` Python package. To serve the TFJS model in a web app (e.g., the `web/` folder), copy the contents of `output_folder/` into your web project's `public/models/` (or similar) and load with the TFJS API.

## Notes & caveats

- `main.py` contains IPython magics (lines starting with `%`). If you run it as a plain script, remove those lines or run it inside a Jupyter environment.
- The script uses `color_mode='grayscale'` and input shape `(100, 100, 1)`. If you change color mode or image size, update the model input layer accordingly.
- The model currently uses `sparse_categorical_crossentropy` and expects integer labels (not one-hot).
- The script uses `ImageDataGenerator` and a manual train/validation split; consider using `image_dataset_from_directory`'s `validation_split` parameter or `tf.data` pipelines for larger datasets.

## Troubleshooting

- If you get an error importing `tensorflow` on macOS with Apple silicon, ensure you installed a compatible TensorFlow build and follow TensorFlow macOS install instructions.
- If `tensorflowjs` conversion fails, verify `tensorflowjs` is installed in the same Python environment and that the `.h5` model file exists and is compatible.

## License

This repository contains example code. Add an appropriate license if you plan to publish it.

---
Small changes recommended: remove IPython magics from `main.py` if you intend to run it as a standalone script. If you'd like, I can create a script-only version of `main.py` (no magics) and a small inference runner—tell me if you want that.
