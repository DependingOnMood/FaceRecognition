from time import time
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score

# Train a SVM-linear kernel classification model
def svm_linear(x_train_data,y_train):
    print("Fitting the classifier to the training set")
    t0 = time()
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],}
    clf = GridSearchCV(SVC(kernel='linear', class_weight='balanced'), param_grid)
    # rbf
    clf = clf.fit(x_train_data, y_train)
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)
    return clf

# Train a SVM-rbf kernel classification model

def svm_rbf(x_train_data,y_train):
    print("Fitting the classifier to the training set")
    t0 = time()
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],}
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    # rbf
    clf = clf.fit(x_train_data, y_train)
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)
    return clf

# Train a Decision Tree classification model
def decisionTree(x_train_data,y_train):
    print("Fitting the classifier to the training set")
    t0 = time()
    clf = DecisionTreeClassifier()
    clf = clf.fit(x_train_data, y_train)
    print("done in %0.3fs" % (time() - t0))
    return clf

# Train a AdaBoost classification model
def adaBoost(x_train_data,y_train):
    print("Fitting the classifier to the training set")
    t0 = time()
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),n_estimators=600,learning_rate=1.5,algorithm="SAMME")
    clf = clf.fit(x_train_data, y_train)
    print("done in %0.3fs" % (time() - t0))
    scores = cross_val_score(clf,x_train_data,y_train)
    print("Scores:")
    print(scores)
    return clf

# Train a KN_N classification model
def KNN(x_train_data,y_train):
    print("Fitting the classifier to the training set")
    t0 = time()
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(x_train_data, y_train)
    print("done in %0.3fs" % (time() - t0))
    scores = cross_val_score(clf,x_train_data,y_train)
    print("Scores:")
    print(scores)
    return clf