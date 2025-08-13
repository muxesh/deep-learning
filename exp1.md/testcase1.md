import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

clf = MLPClassifier(hidden_layer_sizes=(4,), activation='tanh', max_iter=5000, random_state=0)
clf.fit(X, y)

print("Predictions:", clf.predict(X))

for i in range(len(X)):
    color = 'red' if y[i] == 0 else 'blue'
    plt.scatter(X[i][0], X[i][1], color=color)

xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = clf.predict(grid).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.2, cmap='coolwarm')
plt.title('MLP Decision Boundary for XOR')
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.show()

![Screenshot_6-8-2025_122426_colab research google com](https://github.com/user-attachments/assets/93ab40dc-2a90-4046-8afb-f0253d357865)
