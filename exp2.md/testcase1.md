import numpy as np
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
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


sample_indices = [0, 1, 2, 7]  
predicted_labels = model.predict(X_test[sample_indices])
predicted_classes = np.argmax(predicted_labels, axis=1)


true_labels = y_test[sample_indices]
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("Test Cases:")
print("Input Image\tTrue Label\tPredicted Label\tCorrect (Y/N)")
for i, idx in enumerate(sample_indices):
    true = class_names[true_labels[i]]
    pred = class_names[predicted_classes[i]]
    correct = 'Y' if true == pred else 'N'
    print(f"{true}\t{true}\t{pred}\t{correct}")


![Screenshot_13-8-2025_10233_colab research google com](https://github.com/user-attachments/assets/73a66fcb-4630-4d9e-ba69-8501b23a4252)
