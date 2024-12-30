import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, LeakyReLU, BatchNormalization, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from efficientnet_pytorch import EfficientNet
from sklearn.model_selection import train_test_split
import tensorflow_federated as tff
def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_and_preprocess_data()
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)


def create_local_model():
    inputs = Input(shape=(28, 28, 1))
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

local_model = create_local_model()
history = local_model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_val, y_val))


plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Learning Curves')
plt.show()


def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)

def gradient_penalty(discriminator, real_images, fake_images):
    alpha = tf.random.uniform([real_images.shape[0], 1, 1, 1], 0.0, 1.0)
    interpolated = alpha * real_images + (1 - alpha) * fake_images
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = discriminator(interpolated)
    grads = tape.gradient(pred, interpolated)
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    penalty = tf.reduce_mean((norm - 1.0) ** 2)
    return penalty

def build_generator():
    noise = Input(shape=(100,))
    x = Dense(128)(noise)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dense(784, activation='tanh')(x)
    output = Reshape((28, 28, 1))(x)
    return Model(noise, output)

def build_discriminator():
    image = Input(shape=(28, 28, 1))
    x = Flatten()(image)
    x = Dense(128)(x)
    x = LeakyReLU()(x)
    x = Dense(1)(x)
    return Model(image, x)

generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(optimizer=Adam(learning_rate=0.0001), loss=wasserstein_loss)

gen_optimizer = Adam(learning_rate=0.0001)
disc_optimizer = Adam(learning_rate=0.0001)

for epoch in range(10000):
    for _ in range(5):
        real_images = x_train[np.random.randint(0, x_train.shape[0], size=64)]
        noise = np.random.normal(0, 1, (64, 100))
        fake_images = generator.predict(noise)

        with tf.GradientTape() as disc_tape:
            real_pred = discriminator(real_images)
            fake_pred = discriminator(fake_images)
            gp = gradient_penalty(discriminator, real_images, fake_images)
            disc_loss = tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred) + 10.0 * gp
        disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        disc_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

    noise = np.random.normal(0, 1, (64, 100))
    with tf.GradientTape() as gen_tape:
        fake_images = generator(noise)
        gen_pred = discriminator(fake_images)
        gen_loss = -tf.reduce_mean(gen_pred)
    gen_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gen_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Generator Loss: {gen_loss.numpy()}, Discriminator Loss: {disc_loss.numpy()}")


def create_global_model():
    model = EfficientNet.from_name('efficientnet-b7', num_classes=10)
    model._fc = Dense(10, activation='softmax')
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

global_model = create_global_model()
history_global = global_model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))


train_acc = history_global.history['accuracy']
val_acc = history_global.history['val_accuracy']
generalization_gap = [t - v for t, v in zip(train_acc, val_acc)]

plt.plot(generalization_gap, label='Generalization Gap')
plt.xlabel('Epochs')
plt.ylabel('Gap (Train - Validation Accuracy)')
plt.legend()
plt.title('Generalization Gap Across Epochs')
plt.show()


def semantic_embedding(features):
    # Example semantic embedding using feature extraction
    return np.dot(features, np.random.rand(features.shape[1], 128))

def zsl_classification(embeddings, classes, unseen_class_embeddings):
    similarities = np.dot(embeddings, unseen_class_embeddings.T)
    return np.argmax(similarities, axis=1)


feature_extractor = tf.keras.Model(inputs=global_model.input, outputs=global_model.layers[-2].output)
train_features = feature_extractor.predict(x_train)
test_features = feature_extractor.predict(x_test)


unseen_class_embeddings = semantic_embedding(np.random.rand(10, 128))


predicted_classes = zsl_classification(test_features, y_test, unseen_class_embeddings)
print("ZSL Classification completed.")

zsl_accuracy = np.mean(predicted_classes == np.argmax(y_test, axis=1))
plt.bar(['Seen', 'Unseen'], [np.max(history_global.history['val_accuracy']), zsl_accuracy], color=['blue', 'orange'])
plt.ylabel('Accuracy')
plt.title('ZSL Performance: Seen vs Unseen Classes')
plt.show()

def model_fn():
    return tff.learning.from_keras_model(
        keras_model=create_local_model(),
        input_spec=(
            tf.TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 10), dtype=tf.float32)
        ),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()]
    )

iterative_process = tff.learning.build_federated_averaging_process(model_fn)
state = iterative_process.initialize()

def preprocess_fn(dataset):
    def batch_format_fn(element):
        return (tf.expand_dims(element[0], -1) / 255.0, tf.one_hot(element[1], 10))

    return dataset.shuffle(100).batch(20).map(batch_format_fn)

federated_data = [preprocess_fn(tf.data.Dataset.from_tensor_slices((x_train, np.argmax(y_train, axis=1))))]
for round_num in range(10):
    state, metrics = iterative_process.next(state, federated_data)
    print(f"Round {round_num + 1}, Metrics: {metrics}")
