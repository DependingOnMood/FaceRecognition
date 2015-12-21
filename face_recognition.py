from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import data_loader

import photo_resize
import dimensionality_reduction
import classification

# Height and weight of the photo we want to resize to, original photo are mostly 2448*3264 pixels
photo_size =(188, 250)

# number of components(dimensions/eigenfaces)
n_components = 40

# Dimensionality Reduction Type: PCA RandomizedPCA IncrementalPCA ICA NMF
dimensionality_reduction_type = "PCA"
# Classification Type: LinearSVM RBFSVM DecisionTree AdaBoost KNN
classification_type = "LinearSVM"

# Output pages are output_face_row * output_face_col
output_face_row = 3
output_face_col = 3

# Resize the photos to lower resolution for data
photo_resize.resizePhoto(photo_size)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Load the data
people_data = data_loader.getDataBunch(min_faces_per_person=1, resize=1)

# introspect the images arrays to find the shapes (for plotting
n_samples, h, w = people_data.images.shape

# we use the 2 data directly (as relative pixel positions info is ignored by this model)

X = people_data.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = people_data.target
target_names = people_data.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

###############################################################################
# Split into a training and testing set, cross validation, using a stratified k fold
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

###############################################################################
# Choose Dimensionality Reduction Algorithms
# PCA
if dimensionality_reduction_type == "PCA":
    X_train_after_dr, X_test_after_dr, eigenfaces = dimensionality_reduction.compute_PCA(n_components, X_train, X_test,
                                                                                         h, w)
# Randomized PCA
elif dimensionality_reduction_type == "RandomizedPCA":
    X_train_after_dr, X_test_after_dr, eigenfaces = dimensionality_reduction.compute_RandomizedPCA(n_components,
                                                                                                   X_train, X_test, h,
                                                                                                   w)
# IncrementalPCA
elif dimensionality_reduction_type == "IncrementalPCA":
    X_train_after_dr, X_test_after_dr, eigenfaces = dimensionality_reduction.compute_IncrementalPCA(n_components,
                                                                                                    X_train, X_test, h,
                                                                                                    w)
# Fast ICA
elif dimensionality_reduction_type == "ICA":
    X_train_after_dr, X_test_after_dr, eigenfaces = dimensionality_reduction.compute_ICA(n_components, X_train, X_test,
                                                                                         h, w)
# NMF
elif dimensionality_reduction_type == "NMF":
    X_train_after_dr, X_test_after_dr, eigenfaces = dimensionality_reduction.compute_NMF(n_components, X_train, X_test,
                                                                                         h, w)

###############################################################################
# Choose Classification Algorithms
# Linear SVM
if classification_type == "LinearSVM":
    clf = classification.svm_linear(X_train_after_dr, y_train)
# RBF SVM
elif classification_type == "RBFSVM":
    clf = classification.svm_rbf(X_train_after_dr, y_train)
# Decision Tree
elif classification_type == "DecisionTree":
    clf = classification.decisionTree(X_train_after_dr, y_train)
# AdaBoost
elif classification_type == "AdaBoost":
    clf = classification.adaBoost(X_train_after_dr, y_train)
# Kn_n
elif classification_type == "KNN":
    clf = classification.KNN(X_train_after_dr, y_train)

###############################################################################
# Quantitative evaluation of the model quality on the test set

print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_after_dr)
print("done in %0.3fs" % (time() - t0))

print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


###############################################################################
# Qualitative evaluation of the predictions using matplotlib

def plot_gallery(images, titles, h, w, n_row, n_col):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)


prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w, output_face_row, output_face_col)

# plot the gallery of the most significative eigenfaces
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w, output_face_row, output_face_col)

plt.show()
