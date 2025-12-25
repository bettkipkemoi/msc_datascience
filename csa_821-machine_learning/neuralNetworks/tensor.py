# Import necessary libraries (guard TensorFlow imports so script can run
# in environments where TensorFlow is not installed)
import matplotlib.pyplot as plt
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.datasets import fashion_mnist
    from tensorflow.keras.utils import to_categorical
    TF_AVAILABLE = True
except Exception as e:
    tf = None
    TF_AVAILABLE = False
    tf_import_error = e

if TF_AVAILABLE:
    # Load the Fashion MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Preprocess the data
    # Normalize pixel values to be between 0 and 1
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    # Reshape the data to be compatible with the CNN (28, 28, 1)
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

    # One-hot encode the labels
    train_labels = to_categorical(train_labels, num_classes=10)
    test_labels = to_categorical(test_labels, num_classes=10)
else:
    print("TensorFlow not available in this environment.")
    print(f"Import error: {tf_import_error}")
    print("Skipping model training. To run the full script, install TensorFlow in your environment or run inside the `csa-ml` env.")
    # Minimal graceful exit: show a placeholder image
    import numpy as np
    placeholder = np.random.rand(28, 28)
    plt.imshow(placeholder, cmap='gray')
    plt.title('Placeholder image — TensorFlow not available')
    plt.axis('off')
    plt.show()
    raise SystemExit(0)

# Build the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_data=(test_images, test_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\nTest accuracy: {test_acc}')

# Make predictions
predictions = model.predict(test_images)
predicted_classes = tf.argmax(predictions, axis=1)
true_classes = tf.argmax(test_labels, axis=1)

# Display some predictions
num_display = 5
for i in range(num_display):
    plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
    plt.title(f'True: {true_classes[i].numpy()}, Predicted: {predicted_classes[i].numpy()}')
    plt.axis('off')
    plt.show()