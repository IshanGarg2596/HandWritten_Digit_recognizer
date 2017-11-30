import matplotlib.pyplot as plt

from sklearn.externals import joblib
from sklearn import datasets, metrics
from sklearn.svm import LinearSVC
import numpy as np

"""
Two things are needed to be seen:
1. it is taking time to fit 
"""
dataset = datasets.fetch_mldata("MNIST original")

# 10% test 90% train accuracy: 0.84 i.e, 84%
features,labels = np.array(dataset.data[:-7000], 'int16'),np.array(dataset.target[:-7000], 'int')

test_features, test_labels =  np.array(dataset.data[63000:], 'int16'),np.array(dataset.target[63000:], 'int')

clf = LinearSVC()
clf.fit(features, labels)

#calculating accuracy
accuracy = metrics.accuracy_score(test_labels,clf.predict(test_features))

print("accuracy %f"%accuracy)

print(clf.predict(dataset.data[-1000].reshape(1,-1)))

plt.imshow(255-dataset.data[-1000].reshape(28,28),cmap='gray')
plt.show()

joblib.dump(clf, "linearsvm_cls.pkl", compress=3)
