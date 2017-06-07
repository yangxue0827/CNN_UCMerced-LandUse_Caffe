from time import time
from sklearn.metrics import accuracy_score
# from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.svm import SVC

import numpy as np

image_root = '/home/yangxue0827/yangxue/UCMerced_LandUse'

X_train = np.load(image_root + '/train_features.npy')
y_train = np.load(image_root + '/train_labels.npy')
X_test = np.load(image_root + '/test_features1.npy')
y_test = np.load(image_root + '/test_labels1.npy')
# y=y.reshape((-1,1))

X_train = X_train.reshape((-1, 4096))
X_test = X_test.reshape((-1, 4096))
# y_train = y_train.reshape((13340, 1))
# y_test = y_train.reshape((420, 1))
print X_train.shape, y_train.shape, X_test.shape, y_test.shape
# split into a training and testing set
# X_train, _, y_train, _ = train_test_split(
#     X_train, y_train, test_size=0.8, random_state=42)

pca = PCA()
pca.fit(X_train.T)
EVR_List = pca.explained_variance_ratio_
Dim = 0
temp = 0.0
for j in range(len(EVR_List)):
    temp += EVR_List[j]
    if temp >= 0.90:
        Dim = j
        print Dim
        break
pca = PCA(n_components=Dim, copy=True, whiten=False)
X_train = pca.fit(X_train).components_.T
X_test = pca.fit(X_test).components_.T

print("Fitting the classifier to the training set")
t0 = time()
clf = SVC(C=10000, probability=True).fit(X_train, y_train)
print("done in %0.3fs" % (time() - t0))

print("Predicting test set")
t0 = time()
y_pred = clf.predict(X_test)

print "Accuracy: %.3f" % accuracy_score(y_test, y_pred)
joblib.dump(clf, image_root + '/svm1.pkl')

# 测试
# clf1 = joblib.load(image_root + '/svm.pkl')
# pre = clf1.predict(X_test)
# print "Accuracy: %.3f" % accuracy_score(y_test, pre)
