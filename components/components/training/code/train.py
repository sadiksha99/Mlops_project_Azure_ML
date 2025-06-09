import argparse
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from azureml.core import Run
from utils import getFeatures, getTargets, encodeLabels, buildModel

# Constants
SEED = 42
INITIAL_LEARNING_RATE = 0.01
BATCH_SIZE = 32
PATIENCE = 11
MODEL_NAME = "mnist-cnn"

import glob
import os

def collect_image_paths(folder):
    print(f"📂 Scanning folder: {folder}")
    image_paths = glob.glob(os.path.join(folder, "**", "*.jpg"), recursive=True)
    print(f"✅ Found {len(image_paths)} images")
    return image_paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_folder', type=str, required=True)
    parser.add_argument('--testing_folder', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    args = parser.parse_args()

    print("✅ Args:", " ".join(f"{k}={v}" for k, v in vars(args).items()))
    print("📁 Listing training folder contents:", os.listdir(args.training_folder))
    print("📁 Listing testing folder contents:", os.listdir(args.testing_folder))

    # Step 1: Collect images
    training_paths = collect_image_paths(args.training_folder)
    testing_paths = collect_image_paths(args.testing_folder)

    if not training_paths or not testing_paths:
        raise ValueError("❌ No training or testing images found. Ensure folders are named like mnist-2/123.jpg")

    random.seed(SEED)
    random.shuffle(training_paths)
    random.shuffle(testing_paths)

    # Step 2: Extract features and labels
    X_train = getFeatures(training_paths, size=(28, 28), grayscale=True)
    y_train = getTargets(training_paths)

    X_test = getFeatures(testing_paths, size=(28, 28), grayscale=True)
    y_test = getTargets(testing_paths)

    print("📊 Feature shapes:", X_train.shape, X_test.shape)
    print("📊 Label counts:", len(y_train), len(y_test))

    if not y_train or not y_test:
        raise ValueError("❌ Could not extract labels from folder names.")

    LABELS, y_train, y_test = encodeLabels(y_train, y_test)
    print("🏷️ Encoded labels:", LABELS)

    # Step 3: Prepare model output folder
    model_path = os.path.join(args.output_folder, MODEL_NAME)
    os.makedirs(model_path, exist_ok=True)

    # Step 4: Callbacks
    cb_save = keras.callbacks.ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True, verbose=1)
    cb_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1)
    cb_reduce_lr = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, verbose=1)

    # Step 5: Model
    opt = tf.keras.optimizers.legacy.SGD(lr=INITIAL_LEARNING_RATE, decay=INITIAL_LEARNING_RATE / args.epochs)
    model = buildModel((28, 28, 1), len(LABELS))
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # Step 6: Data augmentation
    aug = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    print("🚀 Training...")
    model.fit(
        aug.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_test, y_test),
        steps_per_epoch=max(1, len(X_train) // BATCH_SIZE),
        epochs=args.epochs,
        callbacks=[cb_save, cb_stop, cb_reduce_lr]
    )

    # Step 7: Evaluation
    print("🧪 Evaluating...")
    predictions = model.predict(X_test, batch_size=32)
    print(classification_report(
        y_test.argmax(axis=1),
        predictions.argmax(axis=1),
        target_names=LABELS
    ))

    # Step 8: Save confusion matrix
    cf_matrix = confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
    print("📉 Confusion matrix:\n", cf_matrix)
    np.save(os.path.join(args.output_folder, 'confusion_matrix.npy'), cf_matrix)

    print("✅ Done.")

if __name__ == "__main__":
    main()
