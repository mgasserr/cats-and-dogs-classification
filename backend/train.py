"""
Trains the VGG16 transfer-learning cat/dog classifier, exactly as in the
original notebook, and saves it to model/cat_dog_vgg16.h5 so main.py (the
FastAPI server) can load it.

Requires a Kaggle account for kagglehub to download the dataset the first
time you run this (it will prompt you, or read ~/.kaggle/kaggle.json).

Run with:
    python train.py
"""

import os

import kagglehub
import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "model")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "cat_dog_vgg16.h5")


def main():
    print("Downloading dataset...")
    path = kagglehub.dataset_download("tongpython/cat-and-dog")
    train_dir = os.path.join(path, "training_set", "training_set")
    test_dir = os.path.join(path, "test_set", "test_set")

    print("\n--- Loading Training and Validation Data ---")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )
    print("\n--- Loading Test Data ---")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    print("\n--- Building Model ---")
    base_model = tf.keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    model = models.Sequential(
        [
            layers.Lambda(lambda x: tf.keras.applications.vgg16.preprocess_input(x), input_shape=(224, 224, 3)),
            base_model,
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    print("\n--- Training ---")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    print("\n--- Evaluating ---")
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Final Test Accuracy: {test_acc * 100:.2f}%")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save(OUTPUT_PATH)
    print(f"\nSaved model to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
