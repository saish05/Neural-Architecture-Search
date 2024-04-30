import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import nni

(ds_train, ds_test), ds_info = tfds.load(
    "mnist",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)


def normalize_img(image, label):
    """Normalizes images"""
    return tf.cast(image, tf.float32) / 255.0, label


AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 128

# Setup for train dataset
ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

# Setup for test Dataset
ds_test = ds_test.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.batch(BATCH_SIZE)
ds_test = ds_test.prefetch(AUTOTUNE)


def create_model(params):
    model = keras.Sequential(
        [
            keras.Input((28, 28, 1)),
            layers.Conv2D(params['filter_size_c1'],params['kernel_size_c1'], activation="relu"),
            layers.Conv2D(params['filter_size_c2'],params['kernel_size_c2'], activation= 'relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(params['nb_units'], activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate = params['learning_rate']),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    return model





def run(params) : # Training Loop
    model = create_model(params)
    num_epochs = 1
    for epoch in range(num_epochs):
        print(f"\nStart of Training Epoch {epoch}")
        for batch_idx, (x_batch, y_batch) in enumerate(ds_train):
            loss, train_accuracy = model.train_on_batch(x_batch, y_batch)
            print(f"Batch {batch_idx}, Loss: {loss}, Accuracy: {train_accuracy}")
            
    test_loss, test_accuracy = model.evaluate(ds_test)
    print(f"\nTest Accuracy: {test_accuracy}")
    nni.report_final_result(test_accuracy)

if __name__ == "__main__": 
    params = nni.get_next_parameters()
    run(params)

