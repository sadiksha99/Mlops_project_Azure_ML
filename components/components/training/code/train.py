import argparse
import os
from glob import glob
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
PATIENCE = 10
model_name = 'mnist-cnn'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_folder', type=str, help='Path to training folder')
    parser.add_argument('--testing_folder', type=str, help='Path to testing folder')
    parser.add_argument('--output_folder', type=str, help='Path to output folder')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    args = parser.parse_args()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    training_paths = glob(os.path.join(args.training_folder, "*.jpg"))
    testing_paths = glob(os.path.join(args.testing_folder, "*.jpg"))

    print("Training samples:", len(training_paths))
    print("Testing samples:", len(testing_paths))
    print(training_paths[:3])
    print(testing_paths[:3])

    random.seed(SEED)
    random.shuffle(training_paths)
    random.shuffle(testing_paths)

    X_train = getFeatures(training_paths)
    y_train = getTargets(training_paths)
    X_test = getFeatures(testing_paths)
    y_test = getTargets(testing_paths)

    print("Shapes:")
    print(X_train.shape, X_test.shape, len(y_train), len(y_test))

    LABELS, y_train, y_test = encodeLabels(y_train, y_test)
    print("One-hot encoded shapes:", y_train.shape, y_test.shape)

    model_path = os.path.join(args.output_folder, model_name)
    os.makedirs(model_path, exist_ok=True)

    cb_save_best = keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                   monitor='val_loss',
                                                   save_best_only=True,
                                                   verbose=1)
    cb_early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=PATIENCE,
                                                  restore_best_weights=True)
    cb_reduce_lr = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)

    opt = tf.keras.optimizers.legacy.SGD(lr=INITIAL_LEARNING_RATE,
                                         decay=INITIAL_LEARNING_RATE / args.epochs)

    model = buildModel((28, 28, 1), len(LABELS))
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    aug = ImageDataGenerator(rotation_range=10,
                             zoom_range=0.1,
                             width_shift_range=0.1,
                             height_shift_range=0.1)

    history = model.fit(aug.flow(X_train, y_train, batch_size=BATCH_SIZE),
                        validation_data=(X_test, y_test),
                        steps_per_epoch=len(X_train) // BATCH_SIZE,
                        epochs=args.epochs,
                        callbacks=[cb_save_best, cb_early_stop, cb_reduce_lr])

    print("[INFO] evaluating network...")
    predictions = model.predict(X_test, batch_size=32)
    print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=LABELS))

    cf_matrix = confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
    print(cf_matrix)

    np.save(os.path.join(args.output_folder, "confusion_matrix.npy"), cf_matrix)
    print("DONE TRAINING")


if __name__ == "__main__":
    main()
