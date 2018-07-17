from __future__ import print_function, division
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from astroML.datasets import fetch_sdss_galaxy_colors
from astroML.plotting import scatter_contour
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('SDSS_all.csv') #This function directly queries the sdss SQL database at http://cas.sdss.org/
#data = data[::5]  # truncate for plotting

# put colors in a matrix
def get_features_targets(data):
    features = np.zeros((data.shape[0], 3)) #n lines, 4 columns
    features[:,0] = data['u'] - data['g']
    features[:,1] = data['g'] - data['r']
    features[:,2] = data['redshift']
    #features[:,2] = data['r'] - data['i']
    #features[:,3] = data['i'] - data['z']
    targets = data['specClass']
    return (features, targets)

z = data['redshift']
# Extract colors and spectral class
ug = data['u'] - data['g']
gr = data['g'] - data['r']
spec_class = data['specClass']

stars = (spec_class == 1)
galaxies = (spec_class == 2)
qsos = (spec_class == 3)

fig = plt.figure(1, (12,8))
ax = fig.add_subplot(111)

ax.set_xlim(-0.5, 2.5)
ax.set_ylim(-0.5, 1.5)

ax.plot(ug[galaxies], gr[galaxies], 'o', ms=6, c='g', label='galaxies')
ax.plot(ug[qsos], gr[qsos], 'o', ms=6, c='r', label='qsos')
ax.plot(ug[stars], gr[stars], 'o', ms=6, c='b', label='stars')

ax.legend(loc='upper left', prop={'size':20},frameon=False)
ax.set_xlabel('$u-g$',fontsize=20)
ax.set_ylabel('$g-r$',fontsize=20)

fig = plt.figure(2, (12,8))
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(ug, gr, z, c = spec_class, marker = 'o', s=100)
ax.set_xlabel('$u-g$')
ax.set_ylabel('$g-r$')
ax.set_zlabel('Redshift')

features, targets = get_features_targets(data)
X_train, X_test, y_train, y_test = train_test_split(features, targets,random_state=0)

knn = KNeighborsClassifier(n_neighbors = 4) #choose classifier
KNN_fit = knn.fit(X_train, y_train)#train classifier
accuracy = KNN_fit.score(X_test, y_test) #Estimate the accuracy of the classifier on future data
print ('KNN score: {}\n'.format(accuracy))
cv_scores = cross_val_score(knn, features, targets)
print('Cross-validation scores (3-fold):', cv_scores)
print('Mean cross-validation score (3-fold): {:.3f}'.format(np.mean(cv_scores)))


#t = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
#for s in t:
#    scores = []
#    for i in range(1,1000):
#        X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size = 1-s)
#        knn.fit(X_train, y_train)
#        scores.append(knn.score(X_test, y_test))
#    plt.plot(s, np.mean(scores), 'bo')

#plt.xlabel('Training set proportion (%)')
#plt.ylabel('accuracy')

#Support Vector Machines: unnormalized data
#High C - low regularization (tries to fit training data as well as possible), low C-values - high regularization
svm10RBF = SVC(C=10).fit(X_train, y_train)
svm1RBF = SVC(C=1).fit(X_train, y_train)
svm1 = SVC(kernel='linear', C=1).fit(X_train, y_train)
svm10 = SVC(kernel='linear', C=10).fit(X_train, y_train)
#print('Accuracy of RBF-kernel SVC (C = 1) on training set: {:.2f}'.format(svm1RBF.score(X_train, y_train)),'test set: {:.2f}'.format(svm1RBF.score(X_test, y_test)))
#print('Accuracy of RBF-kernel SVC (C = 10) on training set: {:.2f}'.format(svm10RBF.score(X_train, y_train)),'test set: {:.2f}'.format(svm10RBF.score(X_test, y_test)))
#print('Accuracy of linear-kernel SVC (C = 1) on training set: {:.2f}'.format(svm1.score(X_train, y_train)),'test set: {:.2f}'.format(svm1.score(X_test, y_test)))
#print('Accuracy of linear-kernel SVC (C = 10) on training set: {:.2f}'.format(svm10.score(X_train, y_train)),'test set: {:.2f}'.format(svm10.score(X_test, y_test)))

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() #normalized
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
svm10RBF = SVC(C=10).fit(X_train_scaled, y_train)
svm1RBF = SVC(C=1).fit(X_train_scaled, y_train)
svm1 = SVC(kernel='linear', C=1).fit(X_train_scaled, y_train)
svm10 = SVC(kernel='linear', C=10).fit(X_train_scaled, y_train)
#print('Accuracy of RBF-kernel SVC (C = 1) on normalized training set: {:.2f}'.format(svm1RBF.score(X_train_scaled, y_train)),'test set: {:.2f}'.format(svm1RBF.score(X_test_scaled, y_test)))
print('Accuracy of RBF-kernel SVC (C = 10) on normalized training set: {:.2f}'.format(svm10RBF.score(X_train_scaled, y_train)),'test set: {:.2f}'.format(svm10RBF.score(X_test_scaled, y_test)))
#print('Accuracy of linear-kernel SVC (C = 1) on normalized training set: {:.2f}'.format(svm1.score(X_train_scaled, y_train)),'test set: {:.2f}'.format(svm1.score(X_test_scaled, y_test)))
#print('Accuracy of linear-kernel SVC (C = 10) on normalized training set: {:.2f}'.format(svm10.score(X_train_scaled, y_train)),'test set: {:.2f}'.format(svm10.score(X_test_scaled, y_test)))
svm10RBF.score(X_test_scaled, y_test)
svm_predicted = svm10RBF.predict(X_test_scaled)
confusion = confusion_matrix(y_test, svm_predicted)
print(classification_report(y_test, svm_predicted))

fig = plt.figure(3, (12,8))
import seaborn as sns
sns.heatmap(confusion)
plt.title('RBF Linear Kernel \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, svm_predicted)))
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)

#DecisionTreeClassifier
dtc = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'.format(dtc.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(dtc.score(X_test, y_test)))

tree_predicted = dtc.predict(X_test)
confusion = confusion_matrix(y_test, tree_predicted)
fig = plt.figure(4, (12,8))
sns.heatmap(confusion)
plt.title('Decision Tree classifier \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, tree_predicted)))
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
print(classification_report(y_test, tree_predicted))


# initialize model
#dtr = DecisionTreeRegressor()
# train the model
#dtr.fit(features, targets)
# make predictions using the same features
#predictions = dtr.predict(features)
# print out the first 4 predicted redshifts
#print(predictions[:4])

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
fig = plt.figure(5, (12,8))
ax = fig.add_subplot(111, projection = '3d')

#ax.scatter(ug, gr, z, c = spec_class, marker = 'o', s=100, alpha = 0.5)
ax.set_xlabel('$u-g$')
ax.set_ylabel('$g-r$')
ax.set_zlabel('Redshift')

X, y_true = make_blobs(n_samples=100, centers=3,cluster_std=0.60, random_state=0)

kmeans = KMeans(n_clusters=3)
kmeans.fit(features)
y_kmeans = kmeans.predict(features)
#plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
ax.scatter(centers[:,0],centers[:,1],centers[:,2], marker="o",color='magenta', s=250, linewidths = 10, zorder=10)
#plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);


plt.show()
