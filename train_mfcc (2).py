import numpy as np
import librosa
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import warnings  # <-- 1. IMPORT THE CORRECT WARNINGS LIBRARY

# --- 1. CONFIGURATION ---

# The label column name is 'label'
LABEL_COLUMN_NAME = "label"

# We'll use a standard sample rate for all audio
TARGET_SR = 16000
# Number of MFCC features to extract
N_MFCCS = 13

# The dataset has 5 classes, represented by integers 0, 1, 2, 3, 4.
# (0: andhra_pradesh, 1: gujarat, 2: karnataka, 3: kerala, 4: tamil_nadu)
# We want to load all of them.
LABELS_TO_LOAD = [0, 1, 2, 3, 4]


# --- 2. FEATURE EXTRACTION FUNCTION ---

def extract_mfcc(example):
    """
    Processes one audio example from the dataset.
    1. Loads audio array and resamples.
    2. Extracts MFCCs.
    3. Aggregates MFCCs to a fixed-size vector (mean).
    4. Returns the features and the label.
    """
    audio_data = example["audio"]
    audio_array = audio_data["array"]
    original_sr = audio_data["sampling_rate"]

    # Resample if necessary
    if original_sr != TARGET_SR:
        # <-- 2. FIX: USE `warnings` INSTEAD OF `np.warnings`
        with warnings.catch_warnings(): # Suppress librosa resample warnings
            warnings.simplefilter("ignore")
            audio_array = librosa.resample(y=audio_array, orig_sr=original_sr, target_sr=TARGET_SR)

    # Extract MFCCs
    # Note: n_mfcc in librosa 0.8+, n_cc in older.
    # Using n_mfcc as it's more standard now.
    mfccs = librosa.feature.mfcc(y=audio_array, sr=TARGET_SR, n_mfcc=N_MFCCS)

    # Aggregation: Get a fixed-size vector by taking the mean across time
    mfccs_mean = np.mean(mfccs.T, axis=0)

    # Get the label ID (it's already an integer 0-4)
    label_id = example[LABEL_COLUMN_NAME]

    # Return both as a dictionary
    return {"features": mfccs_mean, "label": label_id}

# --- 3. DATA LOADING AND PROCESSING ---

print("Loading dataset 'DarshanaS/IndicAccentDb' (default split)...")
# Load the 'default' split.
dataset = load_dataset("DarshanaS/IndicAccentDb", split="train", streaming=False)

# --- FOR COLAB: To avoid RAM crash, uncomment the line below ---
# dataset = dataset.select(range(2000))
# ---------------------------------------------------------------

print(f"Original dataset size: {len(dataset)}")

# 1. Filter the dataset to only include the labels we want
print(f"Filtering for labels (integers): {LABELS_TO_LOAD}...")
# This will be slow on one core, but will run.
dataset_filtered = dataset.filter(
    lambda example: example[LABEL_COLUMN_NAME] in LABELS_TO_LOAD
)

# This check might be wrong if you don't load all classes
if len(dataset_filtered) == 0:
    print("\n--- ERROR ---")
    print(f"Filtering still returned 0 samples.")
    print("-------------")
    exit()

print(f"Filtered dataset size: {len(dataset_filtered)}")

# 2. Process all filtered audio files to extract MFCCs
print("Processing audio and extracting MFCCs (this will be slow)...")
processed_dataset = dataset_filtered.map(
    extract_mfcc,
    remove_columns=dataset_filtered.column_names # Remove old columns
)

print("Processing complete.")

# 3. Convert to scikit-learn format (X and y)
X = np.array(processed_dataset["features"])
y = np.array(processed_dataset["label"])

print(f"Feature matrix (X) shape: {X.shape}")
print(f"Labels (y) shape: {y.shape}")


# --- 4. MODEL TRAINING AND EVALUATION ---

if X.shape[0] > 1:
    print("\nStarting model training...")

    # 1. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 2. Scale the features (very important for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Train the SVM classifier
    print("Training SVM classifier...")
    model = SVC(kernel='rbf', C=1.0, random_state=42)
    model.fit(X_train_scaled, y_train)

    # 4. Evaluate the model
    print("Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n--- RESULTS ---")
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("---------------")

else:
    print("Not enough data to train a model.")
