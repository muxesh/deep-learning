import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(X, y)

print("Perceptron Predictions:", clf.predict(X))

for i in range(len(X)):
    color = 'red' if y[i] == 0 else 'blue'
    plt.scatter(X[i][0], X[i][1], color=color)

x_values = [0, 1]
try:
    y_values = -(clf.coef_[0][0] * np.array(x_values) + clf.intercept_) / clf.coef_[0][1]
    plt.plot(x_values, y_values)
except:
    print("Decision boundary could not be plotted.")

plt.title('Perceptron Decision Boundary for XOR')
plt.show()


![Screenshot_6-8-2025_122448_colab research google com](https://github.com/user-attachments/assets/3c1001e9-fb4e-46f6-bc55-f3bf4af9421b)
