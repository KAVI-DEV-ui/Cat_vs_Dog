# 🐱🐶 Cats vs Dogs Classifier (Transfer Learning with MobileNetV2)

*A deep learning project that classifies pet images as cats or dogs using Transfer Learning and MobileNetV2.*

---

## 📌 Project Description

This project builds a **binary image classifier** that distinguishes between **cats and dogs** using **Transfer Learning** with **MobileNetV2** — a powerful and efficient CNN pre-trained on the **ImageNet** dataset.

The dataset, sourced from **TensorFlow Datasets (TFDS)**, contains over **25,000 real-world pet images** of varying sizes and conditions, making it a great benchmark for evaluating model generalization.

---

## 🎯 What This Project Demonstrates

✅ **Loading and preprocessing real-world images** from TensorFlow Datasets
✅ **Building a CNN classifier** using **MobileNetV2** as a **feature extractor**
✅ **Fine-tuning** the base model for improved accuracy
✅ Achieving **90–95% test accuracy** with minimal overfitting
✅ **Making predictions** on new, unseen images

---

## 🧠 Workflow

### 1. **Data Loading & Exploration**

* Import dataset from `tensorflow_datasets` (`cats_vs_dogs`)
* Visualize sample images and label distribution

### 2. **Data Preprocessing**

* Resize images to 160×160 (MobileNetV2 input size)
* Normalize pixel values (0–255 → 0–1)
* Split data into **training**, **validation**, and **test sets**

### 3. **Model Building**

* Load **MobileNetV2** pre-trained on **ImageNet**, without top layers
* Freeze base layers for feature extraction
* Add custom classification head:

  * `GlobalAveragePooling2D`
  * `Dense(128, activation='relu')`
  * `Dense(1, activation='sigmoid')`

### 4. **Training & Fine-Tuning**

* Compile with **Adam optimizer** and **binary crossentropy loss**
* Train on preprocessed images for 10–15 epochs
* Unfreeze top layers for fine-tuning to improve performance

### 5. **Evaluation & Prediction**

* Evaluate model on test dataset
* Predict custom images (cat/dog) with probability scores

---

## 🛠️ Tech Stack

* **Language:** Python 🐍
* **Frameworks & Libraries:**

  * `tensorflow`, `keras` – Deep learning & Transfer Learning
  * `numpy`, `matplotlib` – Data analysis & visualization
  * `tensorflow_datasets` – Dataset loading
  * `PIL` or `OpenCV` – Image loading for predictions

---

## 📂 Repository Structure

```
/cats-vs-dogs-classifier
├──cat_vs_dog_.ipnb
└── README.md
```

---

## ⚙️ How to Run

1. **Clone the Repository**

   ```bash
   git clone https://github.com/KAVI-DEV-ui/cats-vs-dogs-classifier.git
   cd cats-vs-dogs-classifier
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Notebook**

   ```bash
   jupyter notebook cats_dogs_mobilenetv2.ipynb
   ```

4. **Make Predictions on Custom Images**

   ```python
   from tensorflow.keras.models import load_model
   from tensorflow.keras.preprocessing import image
   import numpy as np

   model = load_model('mobilenetv2_model.h5')
   img = image.load_img('sample.jpg', target_size=(160, 160))
   img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
   prediction = model.predict(img_array)
   print("🐶 Dog" if prediction[0][0] > 0.5 else "🐱 Cat")
   ```

---

## 📊 Results

| Model Version                   | Accuracy | Parameters | Notes                            |
| ------------------------------- | -------- | ---------- | -------------------------------- |
| Base CNN (from scratch)         | ~80%     | 3M         | Slower training, lower accuracy  |
| MobileNetV2 (Feature Extractor) | ~92%     | 2.2M       | Fast & efficient                 |
| Fine-Tuned MobileNetV2          | **~95%** | 2.5M       | Best balance of speed & accuracy |

---

## 🚀 Future Enhancements

* 📱 Deploy as a **Streamlit web app** for image uploads
* 🐾 Extend to **multi-class classification** (e.g., breeds)
* 🧠 Experiment with **EfficientNet** or **ResNet50**
* ☁️ Deploy model on **Render / Hugging Face Spaces**

---

## 📄 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Kavi Dev**
GitHub: [KAVI-DEV-ui](https://github.com/KAVI-DEV-ui)
