from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

from sigmoidAFunction import sigmoidActivation
from sigmoidDFunction import sigmoidDerivative


def predict(X, W):
    preds = sigmoidActivation(X.dot(W))

    preds[preds <= 0.5] = 0
    preds[preds > 0] = 1

    return preds


"""Generates a 2-class classification problem with 1000 data points,
   where each data points is a 2D feature vector"""
(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2,
                    cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1))


"""Adds column of 1's as the last entry in the feature matrix, which treats as bias"""
X = np.c_[X, np.ones((X.shape[0]))]

(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.5, random_state=42)

print("[INFO] training...")
W = np.random.randn(X.shape[1], 1)
losses = []

for epoch in np.arange(0, 100):
    preds = sigmoidActivation(trainX.dot(W))

    error = preds - trainY
    loss = np.sum(error ** 2)
    losses.append(loss)

    d = error * sigmoidDerivative(preds)
    gradient = trainX.T.dot(d)

    W += -0.01 * gradient

    if epoch == 0 or (epoch + 1) % 5 ==0:
        print("[INFO] epoch={}, loss={:.7f}".format(int(epoch + 1), loss))

print("[INFO] evaluating...")
preds = predict(testX, W)
print(classification_report(testY, preds))


plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(testX[:, 0], testX[:, 1], marker="o", c=testY[:, 0], s=3)


plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()
