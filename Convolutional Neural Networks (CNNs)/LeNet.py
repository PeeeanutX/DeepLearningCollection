import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the images to the range of 0 to 1
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# Resize the images to add a channel dimension, required by the CNN
train_images = train_images[..., tf.newaxis]
test_images = test_images[..., tf.newaxis]

# Convert the labels to one-hot encoded vectors
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = models.Sequential()

# C1 Convolutional Layer
model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))

# S2 Pooling Layer
model.add(layers.AveragePooling2D(pool_size=(2, 2)))

# C3 Convolutional Layer
model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))

# S4 Pooling Layer
model.add(layers.AveragePooling2D(pool_size=(2, 2)))

# Flatten the output to feed into the DNN
model.add(layers.Flatten())

# C5 Layer
model.add(layers.Dense(units=120, activation='relu'))

# F6 Layer
model.add(layers.Dense(units=84, activation='relu'))

# Output Layer
model.add(layers.Dense(units=10, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, batch_size=128, validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")