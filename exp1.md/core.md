import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])
clf = MLPClassifier(hidden_layer_sizes=(4,), activation='tanh', max_iter=1000, random_state=0)
clf.fit(X, y)
print("Predictions:", clf.predict(X))
for i in range(len(X)):
    if y[i] == 0:
        plt.scatter(X[i][0], X[i][1], color='red')
    else:
        plt.scatter(X[i][0], X[i][1], color='blue')
xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = clf.predict(grid).reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.2, cmap='coolwarm')

plt.title('MLP Decision Boundary for XOR')
plt.show()

![Screenshot_6-8-2025_115836_colab research google com](https://github.com/user-attachments/assets/c09605cb-65a0-4027-864c-5ead4e6fd032)
