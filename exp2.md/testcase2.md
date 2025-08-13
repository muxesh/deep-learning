import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train_cat, epochs=5, batch_size=32, validation_data=(X_test, y_test_cat))

sample_indices = [0, 1, 2, 3] 
predicted_labels = model.predict(X_test[sample_indices])
predicted_classes = np.argmax(predicted_labels, axis=1)
true_labels = y_test[sample_indices]
print("Input Digit Image\tExpected Label\tModel Output\tCorrect (Y/N)")
for i in range(len(sample_indices)):
    expected = true_labels[i]
    predicted = predicted_classes[i]
    correct = 'Y' if expected == predicted else 'N'
    print(f"Image of {expected}\t\t{expected}\t\t{predicted}\t\t{correct}")


![Screenshot_13-8-2025_113057_colab research google com](https://github.com/user-attachments/assets/9e05c470-acfb-42d0-b6bc-b1fb3e4adbab)
