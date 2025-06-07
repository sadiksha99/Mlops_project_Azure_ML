import os
import re
import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Conv2D, MaxPooling2D, Flatten, Activation, Dropout
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from typing import List

def getTargets(filepaths: List[str]) -> List[str]:
    """
    Extract label from filename. Assumes filenames like '4_12345.jpg' or 'img3_999.png'.
    Adjust the regex as needed based on your file naming.
    """
    labels = []
    for fp in filepaths:
        filename = os.path.basename(fp)
        match = re.search(r'\d', filename)  # looks for first digit in filename
        if match:
            labels.append(match.group())
        else:
            raise ValueError(f"No label found in filename: {filename}")
    return labels

def encodeLabels(y_train: List, y_test: List):
    label_encoder = LabelEncoder()
    y_train_labels = label_encoder.fit_transform(y_train)

    try:
        y_test_labels = label_encoder.transform(y_test)
    except ValueError as e:
        print("❌ Label mismatch between train and test sets.")
        print("Train labels:", set(y_train))
        print("Test labels:", set(y_test))
        raise ValueError(f"y contains previously unseen labels: {str(e)}")

    y_train_1h = to_categorical(y_train_labels)
    y_test_1h = to_categorical(y_test_labels)

    LABELS = label_encoder.classes_
    print(f"✔ Encoded labels: {LABELS} -- {label_encoder.transform(LABELS)}")

    return LABELS.tolist(), y_train_1h, y_test_1h

def getFeatures(filepaths: List[str]) -> np.array:
    images = []
    for imagePath in filepaths:
        image = Image.open(imagePath).convert("L").resize((28, 28))  # grayscale
        image = np.array(image, dtype=np.float32) / 255.0            # normalize to [0,1]
        image = np.expand_dims(image, axis=-1)                       # (28, 28, 1)
        images.append(image)
    return np.array(images)

def buildModel(inputShape: tuple, classes: int) -> Sequential:
    model = Sequential()
    height, width, depth = inputShape
    inputShape = (height, width, depth)
    chanDim = -1

    model.add(Conv2D(32, (3, 3), padding="same", name='conv_32_1', input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same", name='conv_64_1'))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding="same", name='conv_64_2'))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding="same", name='conv_128_1'))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same", name='conv_128_2'))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same", name='conv_128_3'))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, name='fc_1'))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(classes, name='output'))
    model.add(Activation("softmax"))

    return model
