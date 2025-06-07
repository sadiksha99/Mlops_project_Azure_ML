import argparse
import os
from glob import glob
import random
import tensorflow as tf
import numpy as np

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

from azureml.core import Run

# Utils: assumed to include getFeatures, getTargets, encodeLabels, buildModel
from utils import *

### HARDCODED VARIABLES
SEED = 42
INITIAL_LEARNING_RATE = 0.01
BATCH_SIZE = 32
PATIENCE = 11
model_name = 'mnist-cnn'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_folder', type=str, dest='training_folder', help='training folder mounting point')
    parser.add_argument('--testing_folder', type=str, dest='testing_folder', help='testing folder mounting point')
    parser.add_argument('--output_folder', type=str, dest='output_folder', help='Output folder')
    parser.add_argument('--epochs', type=int, dest='epochs', help='The amount of Epochs to train')
    args = parser.parse_args()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))
    training_folder = args.training_folder
    testing_folder = args.testing_folder
    output_folder = args.output_folder
    MAX_EPOCHS = args.epochs

    print('Training folder:', training_folder)
    print('Testing folder:', testing_folder)
    print('Output folder:', output_folder)

    training_paths = glob(training_folder + "/*.jpg", recursive=True)
    testing_paths = glob(testing_folder + "/*.jpg", recursive=True)

    print("Training samples:", len(training_paths))
    print("Testing samples:", len(testing_paths))

    random.seed(SEED)
    random.shuffle(training_paths)
    random.seed(SEED)
    random.shuffle(testing_paths)

    print(training_paths[:3])
    print(testing_paths[:3])

    X_train = getFeatures(training_paths)  # expects shape (28, 28, 1)
    y_train = getTargets(training_paths)

    X_test = getFeatures(testing_paths)
    y_test = getTargets(testing_paths)

    print('Shapes:')
    print(X_train.shape, X_test.shape)
    print(len(y_train), len(y_test))

    LABELS, y_train, y_test = encodeLabels(y_train, y_test)
    print('Detected labels:', LABELS)
    print('One Hot Shapes:', y_train.shape, y_test.shape)

    model_path = os.path.join(output_folder, model_name)
    os.makedirs(model_path, exist_ok=True)

    cb_save_best_model = keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                         monitor='val_loss',
                                                         save_best_only=True,
                                                         verbose=1)

    cb_early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=PATIENCE,
                                                  verbose=1,
                                                  restore_best_weights=True)

    cb_reduce_lr_on_plateau = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, verbose=1)

    opt = tf.keras.optimizers.legacy.SGD(
        lr=INITIAL_LEARNING_RATE,
        decay=INITIAL_LEARNING_RATE / MAX_EPOCHS
    )

    model = buildModel((28, 28, 1), len(LABELS))  # adapted for grayscale MNIST
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    aug = ImageDataGenerator(rotation_range=10, width_shift_range=0.1,
                             height_shift_range=0.1, zoom_range=0.1,
                             fill_mode="nearest")

    history = model.fit(
        aug.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_test, y_test),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=MAX_EPOCHS,
        callbacks=[cb_save_best_model, cb_early_stop, cb_reduce_lr_on_plateau]
    )

    print("[INFO] evaluating network...")
    predictions = model.predict(X_test, batch_size=32)
    print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=LABELS))

    cf_matrix = confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
    print(cf_matrix)

    np.save(os.path.join(output_folder, 'confusion_matrix.npy'), cf_matrix)

    print("✅ DONE TRAINING")

if __name__ == "__main__":
    main()
