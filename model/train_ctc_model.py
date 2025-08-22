import tensorflow as tf
from tensorflow.keras import layers, models
from datasets import load_dataset
import numpy as np

# ---- CONFIGURACIÓ ----
img_width, img_height = 128, 32
batch_size = 16
epochs = 5
characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "

# ---- Carregar conjunt de dades ----
dataset = load_dataset("Mir0da/IAM_Handwriting", split="train")
dataset = dataset.rename_column("text", "labels")

# ---- Mapeig de caràcters a índexs ----
char_to_num = tf.keras.layers.StringLookup(vocabulary=list(characters), oov_token="")
num_to_char = tf.keras.layers.StringLookup(vocabulary=list(characters), oov_token="", invert=True)

# ---- Generador de dades ----
def data_generator(dataset, batch_size):
    while True:
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            images = []
            labels = []
            input_lengths = []
            label_lengths = []
            for item in batch:
                img = np.array(item['image'].convert("L").resize((img_width, img_height))) / 255.0
                img = np.expand_dims(img, -1)
                label = item['labels']
                label_encoded = char_to_num(tf.strings.unicode_split(label, input_encoding='UTF-8'))
                images.append(img)
                labels.append(label_encoded)
                input_lengths.append(img.shape[1] // 4)  # Depèn del model
                label_lengths.append(len(label_encoded))
            yield (
                {
                    'image_input': np.array(images),
                    'labels': np.array(labels),
                    'input_length': np.array(input_lengths),
                    'label_length': np.array(label_lengths)
                },
                np.zeros(batch_size)
            )

# ---- Model ----
input_img = layers.Input(shape=(img_height, img_width, 1), name="image_input")
x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(input_img)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2))(x)
new_shape = ((img_width // 4), (img_height // 4) * 128)
x = layers.Reshape(target_shape=new_shape)(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
output = layers.Dense(len(characters) + 1, activation="softmax")(x)

model = models.Model(inputs=input_img, outputs=output)

# ---- CTC Loss ----
def ctc_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    input_length = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])
    label_length = tf.fill([tf.shape(y_true)[0]], tf.shape(y_true)[1])
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

model.compile(optimizer="adam", loss=ctc_loss)

# ---- Entrenament ----
model.fit(
    data_generator(dataset, batch_size),
    steps_per_epoch=len(dataset) // batch_size,
    epochs=epochs
)

# ---- Guardar model ----
model.save("model/ctc_model.h5")
print("✅ Model guardat a model/ctc_model.h5")
