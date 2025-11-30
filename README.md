# 🗣️ Native Language Identification of Indian English Speakers Using HuBERT & MFCCs

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=pytorch\&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FF4B4B.svg?logo=streamlit\&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)

---

## 📌 Project Links

🔗 **GitHub Repository:**
[https://github.com/RohithMB004/voiceAnalysis](https://github.com/RohithMB004/voiceAnalysis)

🔗 **Live Streamlit Demo (Accent-to-Cuisine Recommender):**
[https://rohithmb004-voiceanalysis-app-j4gnxo.streamlit.app/](https://vocalytics-app-j4gnxo.streamlit.app/)

---

## 🎯 Objective

This project aims to classify the **native language (L1)** of Indian speakers by analyzing their **English accent patterns**.
Indian English varies significantly across regions (Malayalam-English, Tamil-English, Kannada-English, Gujarati-English, etc.).
By using **MFCC features** and **HuBERT embeddings**, this project identifies the speaker’s likely linguistic background.

We also built a real-time **Accent-Aware Cuisine Recommender**, which listens to a user’s accent and suggests dishes from their region.

---

## 👥 Team Members

* **Sneha Sooraj**
* **Rohith M B**
* **Ananthu Krishna O**

---

## 📂 Folder Structure

```
voiceAnalysis/
│── app.py                          # Streamlit Application
│── hubert_accent_model_full.pkl    # Trained HuBERT Accent Classifier
│── meow.ipynb                      # HuBERT Layer-wise Analysis Notebook
│── requirements.txt                # Python dependencies
│── train_mfcc.py                   # MFCC-Based Classifier
│── README.md                       # Documentation
└── .gitignore                      # Files to ignore in Git
```

---

## 🗂️ Dataset

We use **IndicAccentDb** from Hugging Face:
[https://huggingface.co/datasets/DarshanaS/IndicAccentDb](https://huggingface.co/datasets/DarshanaS/IndicAccentDb)

Labels included:

| ID | Language / Region |
| -- | ----------------- |
| 0  | Andhra Pradesh    |
| 1  | Gujarat           |
| 2  | Hindi / Jharkhand |
| 3  | Karnataka         |
| 4  | Kerala            |
| 5  | Tamil Nadu        |

---

## 🛠️ Installation & Requirements

### ▶️ 1. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# OR
source venv/bin/activate     # Mac/Linux
```

### ▶️ 2. Install dependencies

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:

```
streamlit
numpy
joblib
librosa
soundfile
torch
torchaudio
transformers
sentencepiece
datasets
scikit-learn
```

---

# 🚀 How to Run Each Component

---

## ▶️ 1. Run the Streamlit Application (Accent-to-Cuisine System)

This is your **main user-facing application**.

### Step 1: Ensure the model file is present

`hubert_accent_model_full.pkl` must be in the **same folder** as `app.py`.

### Step 2: Run Streamlit

```bash
streamlit run app.py
```

### Step 3: Use the App

* Upload a **WAV/MP3** file
* The system extracts HuBERT embeddings
* Predicts the speaker’s **native accent**
* Shows **regional cuisine recommendations**

---

## ▶️ 2. Train the MFCC-Based Accent Classifier

Script: `train_mfcc.py`

### Run:

```bash
python train_mfcc.py
```

This script:

✔ Loads dataset
✔ Extracts MFCCs (13 coefficients)
✔ Mean-pools features
✔ Trains an SVM classifier
✔ Prints accuracy
✔ Generates **real MFCC confusion matrix** and **learning curve** if you added my code

Outputs saved:

* `mfcc_confusion_matrix_real.png`
* `mfcc_learning_curve_real.png`

---

## ▶️ 3. Run HuBERT Layer-wise Model (12-Layer Analysis)

Notebook: `meow.ipynb`

This notebook performs:

1. Load 1000 samples (shuffled)
2. Extract **all 12 HuBERT transformer layers**
3. Train **12 SVM classifiers** (one per layer)
4. Plot **layer-wise accuracy graph**
5. Find **best-performing layer**
6. Generate:

   * Real confusion matrix
   * Real classification report

Saved outputs:

* `hubert_layer_wise_accuracy.png`
* `hubert_confusion_matrix_real.png`

Highest accuracy in your results = **Layer 2 (96.5%)**

---

# 📊 Results Summary

| Model         | Approach         | Accuracy         | Notes                         |
| ------------- | ---------------- | ---------------- | ----------------------------- |
| MFCC-SVM      | Classic features | **90–95%**       | Good baseline                 |
| HuBERT-SVM    | Deep embeddings  | **96–97%**       | Captures accent patterns best |
| Streamlit App | Real-time        | High reliability | Smooth inference              |

---

# 🔮 Future Work

* Scale to **10+ Indian languages**
* Add **live audio recording**
* Explore **Wav2Vec2 / XLSR** embeddings
* Deploy with **GPU acceleration**
* Improve robustness across **children vs adults**

---

# 📜 License

MIT License — free for education, research, and personal use.

---

# 🎉 Acknowledgment

Thanks to **IndicAccentDb creators**, 🤗 **HuggingFace**, and **Facebook AI (HuBERT)**.

This project was created as part of the academic course on speech analytics and machine learning.

---
