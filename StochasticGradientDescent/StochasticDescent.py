import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from sigmoidAFunction import sigmoidActivation
from sigmoidDFunction import sigmoidDerivative


def predict(X, W):
    preds = sigmoidActivation(X.dot(W))

    preds[preds <= 0.5] = 0
    preds[preds > 0] = 1

    return preds


def nextBatch(X, y, batchSize):
    for i in np.arange(0, X.shape[0], batchSize):
        yield X[i:i + batchSize], y[i:i + batchSize]


(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2,
                    cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1))

X = np.c_[np.ones((X.shape[0])), X]

(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.5, random_state=42)

print("[INFO] starting training...")
W = np.random.uniform(size=(X.shape[1], 1))

lossHistory = []


for epoch in np.arange(0, 100):
    epochLoss = []

    for (batchX, batchY) in nextBatch(trainX, trainY, 32):
        preds = sigmoidActivation(batchX.dot(W))

        error = preds - batchY
        loss = np.sum(error ** 2)
        epochLoss.append(loss)

        d= error * sigmoidDerivative(preds)
        gradient = batchX.T.dot(d)

        W += -0.01 * gradient

    loss = np.average(epochLoss)
    lossHistory.append(loss)

    if epoch == 0 or (epoch + 1) % 5 == 0:
        print("[INFO] epoch={}, loss={:.7f}".format(int(epoch + 1), loss))


print("[INFO] evaluating..")
preds = predict(testX, W)
print(classification_report(testY, preds))

plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(testX[:, 0], testX[:, 1], marker="o", c=testY[:, 0], s=30)

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), lossHistory)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()
