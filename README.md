# 🥔 Potato Disease Classification using CNN

A deep learning-based image classification model to detect **Early Blight**, **Late Blight**, and **Healthy** potato leaves using the [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease).

---

## 📌 Key Features

- 🔍 Classifies potato leaves into:
  - **Potato___Early_blight**
  - **Potato___Late_blight**
  - **Potato___healthy**
- 📊 Achieves **~89% test accuracy** after 10 epochs of training
- ⚙️ CNN model built with TensorFlow and Keras
- 📈 Plots training/validation accuracy and loss curves
- 🧪 Includes prediction on test samples with confidence score

---

## 🛠 Tech Stack

- `TensorFlow/Keras` – Model development
- `Matplotlib` – Visualization
- `Python` – General scripting
- `PlantVillage` – Image dataset

---

## 📂 Dataset

- 2,152 labeled images belonging to 3 classes
- Preprocessed and augmented using:
  - Rescaling
  - Random Flip
  - Random Rotation

---

## 🧠 Model Architecture

A sequential CNN model with:

- Multiple `Conv2D + MaxPooling` layers
- `Flatten` and `Dense` layers
- Final output layer with `softmax` activation

Compiled with:

```python
model.compile(
    optimizer='adam',
    loss='SparseCategoricalCrossentropy',
    metrics=['accuracy']
)
```

---

## 📈 Performance

| Epoch | Train Acc | Val Acc | Test Acc |
|-------|-----------|---------|----------|
| 1     | 51.1%     | 50.5%   | –        |
| ...   | ...       | ...     | ...      |
| 10    | 93.3%     | 90.1%   | **89.4%** |

---

## 🔮 Inference

- Predicts disease class from input image
- Shows actual vs predicted labels with **confidence percentage**

Example:
```
Actual: Potato___Early_blight
Predicted: Potato___Early_blight
Confidence: 98.7%
```

---

## 📊 Visualization

- Training/Validation Accuracy and Loss over epochs
- Side-by-side image prediction preview

---

## ▶️ How to Run

```bash
# Install dependencies
pip install tensorflow matplotlib

# Train the model
python potato_classifier.py
```


---

## 🙌 Acknowledgements

- [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- [TensorFlow](https://www.tensorflow.org/)
