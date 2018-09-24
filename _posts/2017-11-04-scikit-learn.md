---
title: Python机器学习 -- Scikit-learn
layout: post
img: scikit-learn.png
tags: [Python, 机器学习]
---
# Scikit-learn

Scikit-learn contains simple and efficient tools for data mining and data analysis.  It implements a wide variety of machine learning algorithms and processes to conduct advanced analytics.

Library documentation: [http://scikit-learn.org/stable/](http://scikit-learn.org/stable/)

### General

**In:**


```python
import numpy as np
from sklearn import datasets
from sklearn import svm

# import a sample dataset and view the data
digits = datasets.load_digits()
print(digits.data)
```

**Out:**

    [[  0.   0.   5. ...,   0.   0.   0.]
     [  0.   0.   0. ...,  10.   0.   0.]
     [  0.   0.   0. ...,  16.   9.   0.]
     ..., 
     [  0.   0.   1. ...,   6.   0.   0.]
     [  0.   0.   2. ...,  12.   0.   0.]
     [  0.   0.  10. ...,  12.   1.   0.]]



**In:**

```python
# view the target variable
digits.target
```

**Out:**


    array([0, 1, 2, ..., 8, 9, 8])


**In:**

```python
# train a support vector machine using everything but the last example 
classifier = svm.SVC(gamma=0.001, C=100.)
classifier.fit(digits.data[:-1], digits.target[:-1])
```

**Out:**


    SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
      gamma=0.001, kernel='rbf', max_iter=-1, probability=False,
      random_state=None, shrinking=True, tol=0.001, verbose=False)



**In:**


```python
# predict the target of the last example
classifier.predict(digits.data[-1])
```

**Out:**


    array([8])


**In:**

```python
# persist the model and reload
import pickle
from sklearn.externals import joblib
joblib.dump(classifier, 'model.pkl')
classifier2 = joblib.load('model.pkl')
classifier2.predict(digits.data[-1])
```

**Out:**


    array([8])


**In:**

```python
import os
os.remove('model.pkl')

# another example with the digits data set
svc = svm.SVC(C=1, kernel='linear')
svc.fit(digits.data[:-100], digits.target[:-100]).score(digits.data[-100:], digits.target[-100:])
```

**Out:**


    0.97999999999999998


**In:**

```python
# perform cross-validation on the estimator's predictions
from sklearn import cross_validation
k_fold = cross_validation.KFold(n=6, n_folds=3)
for train_indices, test_indices in k_fold:
    print('Train: %s | test: %s' % (train_indices, test_indices))
```

**Out:**

    Train: [2 3 4 5] | test: [0 1]
    Train: [0 1 4 5] | test: [2 3]
    Train: [0 1 2 3] | test: [4 5]



**In:**

```python
# apply to the model
kfold = cross_validation.KFold(len(digits.data), n_folds=3)
cross_validation.cross_val_score(svc, digits.data, digits.target, cv=kfold, n_jobs=-1)
```

**Out:**


    array([ 0.93489149,  0.95659432,  0.93989983])


**In:**

```python
# use the grid search module to optimize model parameters
from sklearn.grid_search import GridSearchCV
gammas = np.logspace(-6, -1, 10)
classifier = GridSearchCV(estimator=svc, param_grid=dict(gamma=gammas), n_jobs=-1)
classifier.fit(digits.data[:1000], digits.target[:1000])
```

**Out:**


    GridSearchCV(cv=None,
           estimator=SVC(C=1, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0,
      kernel='linear', max_iter=-1, probability=False, random_state=None,
      shrinking=True, tol=0.001, verbose=False),
           fit_params={}, iid=True, loss_func=None, n_jobs=-1,
           param_grid={'gamma': array([  1.00000e-06,   3.59381e-06,   1.29155e-05,   4.64159e-05,
             1.66810e-04,   5.99484e-04,   2.15443e-03,   7.74264e-03,
             2.78256e-02,   1.00000e-01])},
           pre_dispatch='2*n_jobs', refit=True, score_func=None, scoring=None,
           verbose=0)



**In:**


```python
classifier.best_score_
```

**Out:**


    0.92400000000000004


**In:**

```python
classifier.best_estimator_.gamma
```

**Out:**


    9.9999999999999995e-07


**In:**

```python
# run against the test set
classifier.score(digits.data[1000:], digits.target[1000:])
```

**Out:**


    0.94228356336260977


**In:**

```python
# nested cross-validation example
cross_validation.cross_val_score(classifier, digits.data, digits.target)
```

**Out:**


    array([ 0.93521595,  0.95826377,  0.93791946])
### Other Classifiers

**In:**


```python
# import the iris dataset
iris = datasets.load_iris()

# k nearest neighbors
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(iris.data, iris.target)
```

**Out:**


    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_neighbors=5, p=2, weights='uniform')

**In:**


```python
# decision tree
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(iris.data, iris.target)
```

**Out:**


    DecisionTreeClassifier(compute_importances=None, criterion='gini',
                max_depth=None, max_features=None, max_leaf_nodes=None,
                min_density=None, min_samples_leaf=1, min_samples_split=2,
                random_state=None, splitter='best')

**In:**


```python
# stochastic gradient descent
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(loss="hinge", penalty="l2")
sgd.fit(iris.data, iris.target)
```

**Out:**


    SGDClassifier(alpha=0.0001, class_weight=None, epsilon=0.1, eta0=0.0,
           fit_intercept=True, l1_ratio=0.15, learning_rate='optimal',
           loss='hinge', n_iter=5, n_jobs=1, penalty='l2', power_t=0.5,
           random_state=None, shuffle=False, verbose=0, warm_start=False)

**In:**


```python
# naive bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
print("Number of mislabeled points : %d" % (iris.target != y_pred).sum())
```

**Out:**

    Number of mislabeled points : 6
### Regression

**In:**


```python
# load another sample dataset
diabetes = datasets.load_diabetes()

# linear regression
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(diabetes.data, diabetes.target)
```

**Out:**


    LinearRegression(copy_X=True, fit_intercept=True, normalize=False)


**In:**

```python
# regression coefficients
print(regr.coef_)
```

**Out:**

    [ -10.01219782 -239.81908937  519.83978679  324.39042769 -792.18416163
      476.74583782  101.04457032  177.06417623  751.27932109   67.62538639]



**In:**

```python
# mean squared error
np.mean((regr.predict(diabetes.data)-diabetes.target)**2)
```

**Out:**


    2859.6903987680657


**In:**

```python
# explained variance
regr.score(diabetes.data, diabetes.target)
```

**Out:**


    0.51774942541329338


**In:**

```python
# ridge regression
regr = linear_model.Ridge(alpha=.1)
regr.fit(diabetes.data, diabetes.target)
```

**Out:**


    Ridge(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=None,
       normalize=False, solver='auto', tol=0.001)



**In:**


```python
# lasso regression
regr = linear_model.Lasso()
regr.fit(diabetes.data, diabetes.target)
```

**Out:**


    Lasso(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=1000,
       normalize=False, positive=False, precompute='auto', tol=0.0001,
       warm_start=False)



**In:**


```python
# logistic regression (this is actually a classifier)
iris = datasets.load_iris()
logistic = linear_model.LogisticRegression(C=1e5)
logistic.fit(iris.data, iris.target)
```

**Out:**


    LogisticRegression(C=100000.0, class_weight=None, dual=False,
              fit_intercept=True, intercept_scaling=1, penalty='l2',
              random_state=None, tol=0.0001)

### Preprocessing

**In:**


```python
# feature scaling
from sklearn import preprocessing
X = np.array([[ 1., -1.,  2.],
               [ 2.,  0.,  0.],
               [ 0.,  1., -1.]])
X_scaled = preprocessing.scale(X)

# save the scaling transform to apply to new data later
scaler = preprocessing.StandardScaler().fit(X)
scaler
```

**Out:**


    StandardScaler(copy=True, with_mean=True, with_std=True)


**In:**

```python
scaler.transform(X)
```

**Out:**


    array([[ 0.        , -1.22474487,  1.33630621],
           [ 1.22474487,  0.        , -0.26726124],
           [-1.22474487,  1.22474487, -1.06904497]])



**In:**


```python
# range scaling
min_max_scaler = preprocessing.MinMaxScaler()
X_minmax = min_max_scaler.fit_transform(X)
X_minmax
```

**Out:**


    array([[ 0.5       ,  0.        ,  1.        ],
           [ 1.        ,  0.5       ,  0.33333333],
           [ 0.        ,  1.        ,  0.        ]])



**In:**


```python
# instance normalization using L2 norm
X_normalized = preprocessing.normalize(X, norm='l2')
X_normalized
```

**Out:**


    array([[ 0.40824829, -0.40824829,  0.81649658],
           [ 1.        ,  0.        ,  0.        ],
           [ 0.        ,  0.70710678, -0.70710678]])



**In:**


```python
# category encoding
enc = preprocessing.OneHotEncoder()
enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
enc.transform([[0, 1, 3]]).toarray()
```

**Out:**


    array([[ 1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.]])


**In:**

```python
# binning
binarizer = preprocessing.Binarizer().fit(X)
binarizer.transform(X)
```

**Out:**


    array([[ 1.,  0.,  1.],
           [ 1.,  0.,  0.],
           [ 0.,  1.,  0.]])

### Clustering

**In:**


```python
# k means clustering
from sklearn import cluster
k_means = cluster.KMeans(n_clusters=3)
k_means.fit(iris.data)
```

**Out:**


    KMeans(copy_x=True, init='k-means++', max_iter=300, n_clusters=3, n_init=10,
        n_jobs=1, precompute_distances=True, random_state=None, tol=0.0001,
        verbose=0)

### Decomposition

**In:**


```python
# create a signal with 2 useful dimensions
x1 = np.random.normal(size=100)
x2 = np.random.normal(size=100)
x3 = x1 + x2
X = np.c_[x1, x2, x3]

# compute principal component analysis
from sklearn import decomposition
pca = decomposition.PCA()
pca.fit(X)
```

**Out:**


    PCA(copy=True, n_components=None, whiten=False)


**In:**

```python
pca.explained_variance_
```

**Out:**


    array([  2.77625101e+00,   9.03048616e-01,   3.02456658e-31])


**In:**

```python
# only the 2 first components are useful
pca.n_components = 2
X_reduced = pca.fit_transform(X)
X_reduced.shape
```

**Out:**


```python
(100L, 2L)
```
**In:**

```python
# generate more sample data
time = np.linspace(0, 10, 2000)
s1 = np.sin(2 * time)  # signal 1 : sinusoidal signal
s2 = np.sign(np.sin(3 * time))  # signal 2 : square signal
S = np.c_[s1, s2]
S += 0.2 * np.random.normal(size=S.shape)  # Add noise
S /= S.std(axis=0)  # standardize data

# mix data
A = np.array([[1, 1], [0.5, 2]])  # mixing matrix
X = np.dot(S, A.T)  # generate observations

# compute independent component analysis
ica = decomposition.FastICA()
S_ = ica.fit_transform(X)  # get the estimated sources
A_ = ica.mixing_.T
np.allclose(X,  np.dot(S_, A_) + ica.mean_)
```

**Out:**


    True


