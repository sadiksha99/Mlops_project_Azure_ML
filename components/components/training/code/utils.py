import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Activation, Dropout
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from typing import List
import os

def getTargets(filepaths: List[str]) -> List[str]:
    labels = []
    for path in filepaths:
        # Skip any non-file strings like "INPUT_training_folder"
        if not path.endswith(".jpg") or "mnist-" not in path:
            print(f"⚠️ Skipping invalid path: {path}")
            continue

        folder_name = os.path.basename(os.path.dirname(path))  # e.g. "mnist-2"
        if not folder_name.startswith("mnist-"):
            raise ValueError(f"❌ Could not extract label from: {folder_name}")
        label = folder_name.split('-')[-1]
        labels.append(label)

    if not labels:
        raise ValueError("❌ No valid labels extracted. Are you sure folders are named like mnist-2?")
    return labels


def encodeLabels(y_train: List[int], y_test: List[int]):
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    y_train_1h = to_categorical(y_train_encoded)
    y_test_1h = to_categorical(y_test_encoded)

    LABELS = label_encoder.classes_
    print(f"🏷️ Classes: {LABELS} → {label_encoder.transform(LABELS)}")
    return LABELS, y_train_1h, y_test_1h

def getFeatures(filepaths: List[str], size=(28, 28), grayscale=False) -> np.ndarray:
    images = []
    for imagePath in filepaths:
        image = Image.open(imagePath).convert("L" if grayscale else "RGB")
        image = image.resize(size)
        image = np.array(image)
        if grayscale:
            image = np.expand_dims(image, axis=-1)
        images.append(image)
    return np.array(images)

def buildModel(inputShape: tuple, classes: int) -> Sequential:
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(Dropout(0.1))

    model.add(Dense(classes))
    model.add(Activation("softmax"))
    return model
