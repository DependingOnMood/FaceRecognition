from time import time

from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import NMF

#
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
#
def compute_PCA(n_components, X_train, X_test, h, w):
    print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
    t0 = time()
    pca = PCA(n_components=n_components, whiten=True).fit(X_train)
    print("done in %0.3fs" % (time() - t0))
    eigenfaces = pca.components_.reshape((n_components, h, w))

    print("Projecting the input data on the eigenfaces orthonormal basis")
    t0 = time()
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("done in %0.3fs" % (time() - t0))
    return X_train_pca, X_test_pca, eigenfaces

#
# Compute a Randomized PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
#
def compute_RandomizedPCA(n_components, X_train, X_test, h, w):
    print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
    t0 = time()
    randomizedPca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
    print("done in %0.3fs" % (time() - t0))
    eigenfaces = randomizedPca.components_.reshape((n_components, h, w))

    print("Projecting the input data on the eigenfaces orthonormal basis")
    t0 = time()
    X_train_randomizedPca = randomizedPca.transform(X_train)
    X_test_randomizedPca = randomizedPca.transform(X_test)
    print("done in %0.3fs" % (time() - t0))
    return X_train_randomizedPca, X_test_randomizedPca, eigenfaces

#
# Compute a Incremental PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
#
def compute_IncrementalPCA(n_components, X_train, X_test, h, w):
    print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
    t0 = time()
    incrementalPca = IncrementalPCA(n_components=n_components, whiten=True).fit(X_train)
    print("done in %0.3fs" % (time() - t0))
    eigenfaces = incrementalPca.components_.reshape((n_components, h, w))

    print("Projecting the input data on the eigenfaces orthonormal basis")
    t0 = time()
    X_train_incrementalPca = incrementalPca.transform(X_train)
    X_test_incrementalPca = incrementalPca.transform(X_test)
    print("done in %0.3fs" % (time() - t0))
    return X_train_incrementalPca, X_test_incrementalPca, eigenfaces

#
# Compute a ICA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
#
def compute_ICA(n_components, X_train, X_test, h, w):
    print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
    t0 = time()
    ica = FastICA(n_components=n_components, whiten=True).fit(X_train)
    print("done in %0.3fs" % (time() - t0))
    eigenfaces = ica.components_.reshape((n_components, h, w))

    print("Projecting the input data on the eigenfaces orthonormal basis")
    t0 = time()
    X_train_ica = ica.transform(X_train)
    X_test_ica = ica.transform(X_test)
    print("done in %0.3fs" % (time() - t0))
    return X_train_ica, X_test_ica, eigenfaces

#
# Compute a NMF (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
#
def compute_NMF(n_components, X_train, X_test, h, w):
    print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
    t0 = time()
    nmf = NMF(n_components= n_components).fit(X_train)
    print("done in %0.3fs" % (time() - t0))
    eigenfaces = nmf.components_.reshape((n_components, h, w))

    print("Projecting the input data on the eigenfaces orthonormal basis")
    t0 = time()
    X_train_nmf = nmf.transform(X_train)
    X_test_nmf = nmf.transform(X_test)
    print("done in %0.3fs" % (time() - t0))
    return X_train_nmf, X_test_nmf, eigenfaces
