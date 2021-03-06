from sklearn.externals import joblib
from sklearn import datasets

import numpy as np

import matplotlib.pyplot as plt

clf = joblib.load("knn_cls.pkl")

dataset = datasets.fetch_mldata("MNIST Original")

print(clf.predict(dataset.data[-30000].reshape(1,-1)))

plt.imshow(255-dataset.data[-30000].reshape(28,28),cmap='gray')
plt.show()