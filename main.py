import tensorflow as tf
import tensorflow_datasets as tfds
import valohai

# Valohai:
# We define a step called train-model
# and 2 parameters (epochs and learning_rate)
valohai.prepare(
    step="train-model",
    image="tensorflow/tensorflow:2.12.0-gpu",
    default_parameters={"learning-rate": 0.001, "epochs": 10},
)

# Valohai:
# Read the actual parameter values passed to the job
# The valohai.prepare() contains only the default values
epochs = valohai.parameters("epochs").value
lr = valohai.parameters("learning-rate").value


# Valohai:
# We'll call this method at the end of every training epoch
# It will print out the current epoch, accuracy, and loss
def log_metadata(epoch, logs):
    with valohai.logger() as logger:
        logger.log("epoch", epoch)
        logger.log("val_loss", logs["val_loss"])
        logger.log(
            "val_sparse_categorical_accuracy", logs["val_sparse_categorical_accuracy"]
        )


# Below starts the standard TF example
# https://www.tensorflow.org/datasets/keras_example
#
# Load the sample dataset using Tensorflow datasets
(ds_train, ds_test), ds_info = tfds.load(
    valohai.inputs("dataset"),
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)


# Method to apply standard transformations
def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255.0, label


# Define the training
ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

# Define the testing
ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)


# Create and train the model
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10),
    ]
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

# Valohai:
# Adding a custom callback function to call our log_metadata method
# at the end of each epoch
callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_metadata)
model.fit(ds_train, epochs=epochs, validation_data=ds_test, callbacks=[callback])

# Valohai:
# We save the trained model to the outputs
model_path = valohai.outputs().path("model.h5")
model.save(model_path)
