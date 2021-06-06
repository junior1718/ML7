from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from IPython.display import Image
from imutils import paths
import numpy as np
import cv2
import os


def extract_histogram(image, bins=(8, 8, 8)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


imagePaths = sorted(list(paths.list_images('train')))
data = []
labels = []

for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath, 1)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]
    hist = extract_histogram(image)
    data.append(hist)
    labels.append(label)

le = LabelEncoder()
labels = le.fit_transform(labels)
print(labels[0])

# Image(filename=imagePaths[0])
# img = cv2.imread(imagePaths[0], 0)
# cv2.imshow('', img)
# cv2.waitKey(0)

(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(data), labels, test_size=0.25,
                                                                  random_state=6)

model = LinearSVC(random_state=6, C=1.39)
model.fit(trainData, trainLabels)

predictions = model.predict(testData)

print(classification_report(testLabels, predictions, target_names=le.classes_))


predictions = model.predict(testData)
f1_score(testLabels, predictions, average='macro')

print('эта 224 = ', model.coef_[0][224])
print('эта 195 = ', model.coef_[0][195])
print('эта 252 = ', model.coef_[0][252])

singleImage = cv2.imread('test/cat.1017.jpg')
histt = extract_histogram(singleImage)
histt2 = histt.reshape(1, -1)
prediction = model.predict(histt2)

print('cat1017', prediction)

singleImage = cv2.imread('test/cat.1034.jpg')
histt = extract_histogram(singleImage)
histt2 = histt.reshape(1, -1)
prediction = model.predict(histt2)

print('cat1034', prediction)

singleImage = cv2.imread('test/dog.1010.jpg')
histt = extract_histogram(singleImage)
histt2 = histt.reshape(1, -1)
prediction = model.predict(histt2)

print('dog.1010', prediction)

singleImage = cv2.imread('test/cat.1000.jpg')
histt = extract_histogram(singleImage)
histt2 = histt.reshape(1, -1)
prediction = model.predict(histt2)

print('cat.1000', prediction)



