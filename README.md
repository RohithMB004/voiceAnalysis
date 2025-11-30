---

# 🗣️ Native Language Identification of Indian English Speakers Using HuBERT & MFCCs

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=pytorch\&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FF4B4B.svg?logo=streamlit\&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)

---

## 🎯 Objective

This project aims to classify the **native language (L1)** of Indian speakers by analyzing the way they speak **English**. Since Indian languages influence English pronunciation differently—Malayalam-English, Tamil-English, Kannada-English, Gujarati-English, etc.—accent patterns can be used to automatically predict the speaker’s linguistic background.

The project uses two major approaches:

* 🎵 **MFCC-based acoustic features** (traditional audio processing)
* 🧠 **HuBERT-based deep speech embeddings** (self-supervised learning)

We evaluate the performance of both approaches and deploy the HuBERT-based model in a real-world application: a fun **Accent-Aware Cuisine Recommender** that suggests regional foods based on a user’s accent.

---

## 👥 Team Members

* **Sneha Sooraj**
* **Rohith M B**
* **Ananthu Krishna O**

---

## 📦 Dataset

The project uses the **IndicAccentDb** dataset from Hugging Face, which includes recordings of Indian speakers from different native-language backgrounds.

📌 **Dataset Link:**
[https://huggingface.co/datasets/DarshanaS/IndicAccentDb](https://huggingface.co/datasets/DarshanaS/IndicAccentDb)

Each audio file is labeled with one of the following L1 categories:

| Label | Native Language / Region |
| ----- | ------------------------ |
| 0     | Andhra Pradesh           |
| 1     | Gujarat                  |
| 2     | Jharkhand / Hindi        |
| 3     | Karnataka                |
| 4     | Kerala                   |
| 5     | Tamil Nadu               |

Your Streamlit app uses 6 classes, and the MFCC training script uses 5 (based on dataset split).

---

## ⚙️ Project Components

### **1️⃣ MFCC Feature-Based Model**

Your script `train_mfcc.py` extracts:

* 13-dimensional MFCC features
* Mean-pooled vectors for fixed-length embeddings
* Trains an SVM classifier (RBF kernel)
* Standard scaling applied before classification

✔ Demonstrates the performance of traditional handcrafted features
✔ Simple and efficient baseline model

---

### **2️⃣ HuBERT Deep Embedding Model**

Using the file `hubert_accent_model_full.pkl`, the app loads:

* ⚡ **HuBERT-base (facebook/hubert-base-ls960)**
* Extracts contextual speech embeddings
* Mean pooling of last hidden layer
* Feeding scaled embeddings into an ML classifier
* Produces region predictions with high confidence

🎯 HuBERT captures deeper phonetic cues → better accent recognition.

---

### **3️⃣ Accent-Aware Cuisine Recommender App (Streamlit)**

File: `app.py`

The UI allows users to:

1. Upload or record a short English audio clip (WAV/MP3)
2. System extracts HuBERT embeddings
3. Machine learning classifier predicts native accent
4. A cuisine from that region is recommended

#### Example Output:

| Detected Accent | Region            | Suggested Dishes            |
| --------------- | ----------------- | --------------------------- |
| Kerala          | Malayalam-English | Appam, Puttu, Karimeen Fry  |
| Karnataka       | Kannada-English   | Bisi Bele Bath, Mysore Dosa |
| Andhra Pradesh  | Telugu-English    | Gongura Pachadi             |

This makes the system intuitive, fun, and culturally relevant.

---

## 🧠 Technical Workflow

### **1. Feature Extraction**

#### MFCC Pipeline:

* Load → Resample → Compute MFCC → Mean Aggregation

#### HuBERT Pipeline:

* Raw audio → Wav2Vec2FeatureExtractor → HuBERT Model → Embedding → Scaling

---

### **2. Classification Models**

| Feature | Model               | Notes                              |
| ------- | ------------------- | ---------------------------------- |
| MFCC    | SVM                 | Good baseline, interpretable       |
| HuBERT  | SVM / ML classifier | More accurate due to deep features |

---

### **3. Deployment**

The Streamlit UI (`app.py`):

* Uses cached loading for faster execution
* Real-time audio processing
* Recommends region-specific dishes

---

## 📊 Results Summary

| Experiment        | Approach          | Accuracy         | Insights                              |
| ----------------- | ----------------- | ---------------- | ------------------------------------- |
| MFCC Model        | SVM               | ~90–95% (approx) | MFCC captures basic phonetic features |
| HuBERT Embeddings | ML classifier     | ~98–99%          | Better accent discrimination          |
| Streamlit App     | Real-time testing | High reliability | Good real-world performance           |

---

## 🛠️ Tech Stack

| Category         | Tools                       |
| ---------------- | --------------------------- |
| ML Framework     | PyTorch, scikit-learn       |
| Audio Processing | Librosa, HuBERT             |
| Deployment       | Streamlit                   |
| Data             | IndicAccentDb               |
| Others           | NumPy, Transformers, Joblib |

---

## 🚀 Future Enhancements

* Expand to 10+ Indian languages
* Add direct speech recording instead of upload
* Improve cross-age performance (adult → child generalization)
* Deploy on cloud with GPU inference

---

## 📁 Repository Structure

```
project/
│── app.py                       # Streamlit frontend
│── train_mfcc.py                # MFCC training code
│── hubert_accent_model_full.pkl # Trained HuBERT model
│── README.md                    # Project documentation
│── data/                        # Audio samples (optional)
```

---

## 📜 License

This project is released under the **MIT License**.

---

## 🎉 Acknowledgment

This project was developed as part of an academic exploration into speech processing, machine learning, and regional accent modeling in India.

---
