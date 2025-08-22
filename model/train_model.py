import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds

# Carregar EMNIST Balanced via tensorflow-datasets
ds_train, ds_test = tfds.load(
    'emnist/balanced',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True
)

# Funció de preprocessament
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, -1)  # afegim canal
    return image, label

ds_train = ds_train.map(preprocess).batch(128).prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.map(preprocess).batch(128).prefetch(tf.data.AUTOTUNE)

num_classes = 47  # EMNIST Balanced té 47 classes

# Model CNN senzill
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(ds_train, epochs=5, validation_data=ds_test)

model.save("model/model.h5")
print("✅ Model guardat a model/model.h5")
