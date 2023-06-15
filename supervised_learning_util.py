import numpy as np
import sklearn as sk
from matplotlib import pyplot as plt
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import *
from sklearn import tree
from sklearn import ensemble
from sklearn import neural_network
from sklearn import naive_bayes
from sklearn import discriminant_analysis
from sklearn import metrics
from sklearn import inspection
from sklearn.model_selection import GridSearchCV
import os
from g2v_util import TSNE_2D_plot
import time
from datetime import timedelta

def generate_k_nearest_neighbors(X, y, X_test, y_test):
    #n_neighbors = [5, 10, 15, 20, 25, 30]
    #n_neighbors = [30, 35, 45 ]
    n_neighbors = [3, 5, 30 ] #, 45, 60, 100]
    #weights = ['uniform', 'distance']
    weights = ['distance']
    algorithm = ['ball_tree']   #, 'kd_tree', 'brute', 'auto' ]
    leaf_size = [5, 10, 15 ] #, 20, 25, 30, 35, 40, 45]# try smaller than 15 leaf size
    #leaf_size = [5, 50, 80]
    p = [ 1, 1.5, 2, 2.2, 2.5, 2.7, 3.0]
    metric = ['minkowski']  # , 'cityblock', 'cosine', 'euclidean', 'haversine',
    # 'manhattan', 'nan_euclidean' ]
    # n_jobs = [1, -1]
    
    model_preds = []
    best_rec = ()
    pred_best = 0
    curr_score = 0
    
    for i in n_neighbors:
        for j in p:
            for k in leaf_size:
                model = sk.neighbors.KNeighborsClassifier(n_neighbors=i,
                                                          weights='distance',
                                                          algorithm='ball_tree',
                                                          leaf_size=k,
                                                          p=j).fit(X, y)
                y_pred = model.predict(X_test)
                curr_score = sk.metrics.accuracy_score(y_test, y_pred)
                model_preds.append((curr_score, i, "weights='distance'", "algorithm='ball_tree'", k, j))
                if curr_score > pred_best:
                    best_rec = (curr_score, i, "weights='distance'", "algorithm='ball_tree'", k, j)
                    model1 = model
                    pred_best = curr_score  
    '''
    for i in n_neighbors:
        for j in weights:
            for n in p:
                for k in algorithm:
                    if k == 'ball_tree' or k == 'kd_tree':
                        for m in leaf_size:
                            model = sk.neighbors.KNeighborsClassifier(n_neighbors=i,
                                                                      weights=j,
                                                                      algorithm=k,
                                                                      leaf_size=m,
                                                                      p=n)
                            model.fit(X, y)
                            curr_score = model.score(X_test, y_test)
                            model_preds.append((curr_score, i, j, k, m, n))
                            if curr_score > pred_best:
                                best_rec = (curr_score, i, j, k, m, n)
                                model1 = model
                                pred_best = curr_score
                    else:
                        model = sk.neighbors.KNeighborsClassifier(n_neighbors=i,
                                                                  weights=j,
                                                                  algorithm=k,
                                                                  p=n)
                        model.fit(X, y)
                        curr_score = model.score(X_test, y_test)
                        model_preds.append((curr_score, i, j, k, n))
                        if curr_score > pred_best:
                            best_rec = (curr_score, i, j, k, n)
                            model1 = model
                            pred_best = curr_score
    '''
    '''
    parameters = {  'n_neighbors' : ( 3, 5, 30 ),
                    'weights' : ['distance'],
                    'algorithm' : ['ball_tree'],
                    'leaf_size' :  ( 5, 10, 15 ), 
                    'p' : ( 1, 2, 2.2, 2.5, 2.7, 3.0 ) }
    knn = sk.neighbors.KNeighborsClassifier()
    model1 = GridSearchCV(estimator=knn, param_grid=parameters)
    
    model1.fit(X, y)
    #best_rec = model1.get_params()
    '''
    print(model_preds)
    return model1, best_rec


def generate_radius_nearest_neighbors(X, y, X_test, y_test):
    radius = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    weights = ['uniform', 'distance']
    algorithm = ['ball_tree', 'kd_tree', 'brute']
    leaf_size = [15, 20, 25, 30, 35, 40, 45]
    p = [1, 2]
    model_preds = []
    best_rec = ()
    pred_best = 0
    curr_score = 0
    for i in radius:
        for j in weights:
            for m in p:
                for k in algorithm:
                    if k == 'ball_tree' or k == 'kd_tree':
                        for l in leaf_size:
                            model = sk.neighbors.RadiusNeighborsClassifier(radius=i,
                                                                           weights=j,
                                                                           algorithm=k,
                                                                           leaf_size=l,
                                                                           p=m,
                                                                           outlier_label=1)
                            model.fit(X, y)
                            curr_score = model.score(X_test, y_test)
                            model_preds.append((curr_score, i, j, k, l, m))
                            if curr_score > pred_best:
                                best_rec = (curr_score, i, j, k, l, m)
                                model1 = model
                                pred_best = curr_score
                    else:
                        model = sk.neighbors.RadiusNeighborsClassifier(radius=i,
                                                                       weights=j,
                                                                       algorithm=k,
                                                                       p=m,
                                                                       outlier_label='most_frequent')
                        model.fit(X, y)
                        curr_score = model.score(X_test, y_test)
                        model_preds.append((curr_score, i, j, k, m))
                        if curr_score > pred_best:
                            best_rec = (curr_score, i, j, k, m)
                            model1 = model
                            pred_best = curr_score
    #print(best_rec)
    return model1, best_rec


def generate_linear_svc(X, y, X_test, y_test):
    penalty = ['l1', 'l2']
    loss = ['hinge', 'squared_hinge']
    dual = [True, False]
    tol = [.001, 0.0001, .00001]
    C = [.5, .6, .7, .8, .9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    # multi_class='ovr'
    # fit_intercept=True1
    # intercept_scaling=1
    class_weight = None
    # verbose=0
    random_state = [1, 2, 3, 4, 5]
    #max_iter = [1000, 2000, 3000]

    model_preds = []
    best_rec = ()
    pred_best = 0
    curr_score = 0

    for i in penalty:
        for j in loss:
            if i == 'l1' and j == 'hinge':
                continue
            for m in tol:
                for n in C:
                    for k in dual:
                        if k == True and not (i == 'l1' and j == 'squared_hinge'):
                            for o in random_state:
                                model = sk.svm.LinearSVC(penalty=i,
                                                         loss=j,
                                                         dual=k,
                                                         tol=m,
                                                         C=n,
                                                         random_state=o)
                                model.fit(X, y)
                                curr_score = model.score(X_test, y_test)
                                if curr_score > pred_best:
                                    model1 = model
                                    pred_best = curr_score
                                    best_rec = (curr_score, i, j, k, m, n, o)
                        elif k == False and not (i == 'l2', j == 'hinge'):
                            model = sk.svm.LinearSVC(penalty=i,
                                                     loss=j,
                                                     dual=k,
                                                     tol=m,
                                                     C=n)
                            model.fit(X, y)
                            curr_score = model.score(X_test, y_test)
                            if curr_score > pred_best:
                                model1 = model
                                pred_best = curr_score
                                best_rec = (curr_score, i, j, k, m, n)

    #print(best_rec)
    return model1, best_rec


def generate_nu_svc(X, y, X_test, y_test):
    nu = [.1, .2, .3, .4, .5]
    # kernel= [ 'linear', 'rbf', 'sigmoid', 'precomputed', 'poly' ]
    kernel = ['linear', 'rbf', 'sigmoid', 'poly']
    degree = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # for use only with poly kernel
    gamma = ['scale', 'auto']  # for use only with rbf, poly and sigmoid
    coef0 = [0.0, .5, 1.0]  # for use only with poly and sigmoid
    shrinking = [True, False]
    probability = [True, False]
    tol = [.01, 0.001, .0001]
    cache_size = 200
    class_weight = None
    #max_iter = [1000, 2000, 3000, 4000]
    # decision_function_shape='ovr', #for use with multiclass classification
    # break_ties=False #for use with decision_function_shape
    # random_state=None #ignored when probability is false

    model_preds = []
    best_rec = ()
    pred_best = 0
    curr_score = 0

    for i in nu:
        for j in shrinking:
            for l in tol:
                for m in probability:
                    for k in kernel:
                        if k == 'poly':
                            for n in gamma:
                                for o in coef0:
                                    for p in degree:
                                        model = sk.svm.NuSVC(nu=i,
                                                             kernel=k,
                                                             degree=p,
                                                             gamma=n,
                                                             coef0=o,
                                                             shrinking=j,
                                                             probability=m,
                                                             tol=l,
                                                             max_iter = 10000)
                                        model.fit(X, y)
                                        curr_score = model.score(X_test, y_test)
                                        model_preds.append((curr_score, i, k, p, n, o, j, m, l))
                                        if curr_score > pred_best:
                                            model1 = model
                                            pred_best = curr_score
                                            best_rec = (curr_score, i, k, p, n, o, j, m, l)

                        elif k == 'sigmoid':
                            for n in gamma:
                                for o in coef0:
                                    model = sk.svm.NuSVC(nu=i,
                                                         kernel=k,
                                                         gamma=n,
                                                         coef0=o,
                                                         shrinking=j,
                                                         probability=m,
                                                         tol=l,
                                                         max_iter=10000)
                                    model.fit(X, y)
                                    curr_score = model.score(X_test, y_test)
                                    model_preds.append((curr_score, i, k, n, o, j, m, l))
                                    if curr_score > pred_best:
                                        model1 = model
                                        pred_best = curr_score
                                        best_rec = (curr_score, i, k, n, o, j, m, l)
                        elif k == 'rbf':
                            for n in gamma:
                                model = sk.svm.NuSVC(nu=i,
                                                     kernel=k,
                                                     gamma=n,
                                                     shrinking=j,
                                                     probability=m,
                                                     tol=l,
                                                     max_iter=10000)
                                model.fit(X, y)
                                curr_score = model.score(X_test, y_test)
                                model_preds.append((curr_score, i, k, n, j, m, l))
                                if curr_score > pred_best:
                                    model1 = model
                                    pred_best = curr_score
                                    best_rec = (curr_score, i, k, n, j, m, l)
                        else:
                            model = sk.svm.NuSVC(nu=i,
                                                 kernel=k,
                                                 shrinking=j,
                                                 probability=m,
                                                 tol=l,
                                                 max_iter = 10000)
                            # print (len(X[1]), len(y[1]))
                            model.fit(X, y)
                            curr_score = model.score(X_test, y_test)
                            model_preds.append((curr_score, i, k, j, m, l))
                            if curr_score > pred_best:
                                model1 = model
                                pred_best = curr_score
                                best_rec = (curr_score, i, k, j, m, l)
    #print(best_rec)
    return model1, best_rec


def generate_svc(X, y, X_test, y_test):
    C = [.1, 1, 10, 100, 1000]
    # kernel= [ 'linear', 'rbf', 'sigmoid', 'precomputed', 'poly' ]
    kernel = ['linear', 'rbf', 'sigmoid', 'poly']
    degree = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # for use only with poly kernel
    gamma = ['scale', 'auto']  # for use only with rbf, poly and sigmoid
    coef0 = [0.0, .5, 1.0]  # for use only with poly and sigmoid
    shrinking = [True, False]
    probability = [True, False]
    tol = [.01, 0.001, .0001]
    cache_size = 200
    class_weight = None
    #max_iter = [1000, 2000, 3000, 4000]
    # decision_function_shape='ovr', #for use with multiclass classification
    # break_ties=False #for use with decision_function_shape
    # random_state=None #ignored when probability is false

    model_preds = []
    best_rec = ()
    pred_best = 0
    curr_score = 0

    for i in C:
        for j in shrinking:
            for l in tol:
                for m in probability:
                    for k in kernel:
                        if k == 'poly':
                            for n in gamma:
                                for o in coef0:
                                    for p in degree:
                                        model = sk.svm.SVC(C=i,
                                                           kernel=k,
                                                           degree=p,
                                                           gamma=n,
                                                           coef0=o,
                                                           shrinking=j,
                                                           probability=m,
                                                           tol=l,
                                                           max_iter = 10000)
                                        model.fit(X, y)
                                        curr_score = model.score(X_test, y_test)
                                        model_preds.append((curr_score, i, k, p, n, o, j, m, l))
                                        if curr_score > pred_best:
                                            model1 = model
                                            pred_best = curr_score
                                            best_rec = (curr_score, i, k, p, n, o, j, m, l)

                        elif k == 'sigmoid':
                            for n in gamma:
                                for o in coef0:
                                        model = sk.svm.SVC(C=i,
                                                           kernel=k,
                                                           gamma=n,
                                                           coef0=o,
                                                           shrinking=j,
                                                           probability=m,
                                                           tol=l,
                                                           max_iter = 10000)
                                        model.fit(X, y)
                                        curr_score = model.score(X_test, y_test)
                                        model_preds.append((curr_score, i, k, n, o, j, m, l))
                                        if curr_score > pred_best:
                                            model1 = model
                                            pred_best = curr_score
                                            best_rec = (curr_score, i, k, n, o, j, m, l)
                        elif k == 'rbf':
                            for n in gamma:
                                model = sk.svm.SVC(C=i,
                                                   kernel=k,
                                                   gamma=n,
                                                   shrinking=j,
                                                   probability=m,
                                                   tol=l,
                                                   max_iter = 10000)
                                model.fit(X, y)
                                curr_score = model.score(X_test, y_test)
                                model_preds.append((curr_score, i, k, n, j, m, l))
                                if curr_score > pred_best:
                                    model1 = model
                                    pred_best = curr_score
                                    best_rec = (curr_score, i, k, n, j, m, l)
                        else:
                            model = sk.svm.SVC(C=i,
                                               kernel=k,
                                               shrinking=j,
                                               probability=m,
                                               tol=l,
                                               max_iter = 10000)
                            # print (len(X[1]), len(y[1]))
                            model.fit(X, y)
                            curr_score = model.score(X_test, y_test)
                            model_preds.append((curr_score, i, k, j, m, l))
                            if curr_score > pred_best:
                                model1 = model
                                pred_best = curr_score
                                best_rec = (curr_score, i, k, j, m, l)
    #print(best_rec)
    return model1, best_rec


def generate_gaussian_process(X, y, X_train, y_train):
    # kernel= [ 'linear', 'rbf', 'sigmoid', 'poly' ]
    optimizer = ['fmin_l_bfgs_b', None]
    n_restarts_optimizer = [0, 5, 10, 100]
    max_iter_predict = [100, 1000, 1000]
    warm_start = [True, False]
    copy_X_train = True  # playing with this is not likely necessary
    random_state = None  # playing with this is not likely necessary
    multi_class = 'one_vs_rest'  # for use with multiclass classification
    n_jobs = None  # for use with parallel environments

    model_preds = []
    best_rec = ()
    pred_best = 0
    curr_score = 0

    # for i in kernel:
    for j in optimizer:
        for k in n_restarts_optimizer:
            for l in warm_start:
                for m in max_iter_predict:
                    model = sk.gaussian_process.GaussianProcessClassifier(  # kernel = i,
                        optimizer=j,
                        n_restarts_optimizer=k,
                        warm_start=l,
                        max_iter_predict=m)
                    model.fit(X, y)
                    curr_score = model.score(X_train, y_train)
                    model_preds.append((curr_score, j, k, l, m))
                    if curr_score > pred_best:
                        pred_best = curr_score
                        best_rec = (curr_score, j, k, l, m)
    #print(best_rec)
    return model1, best_rec


def generate_decision_tree(X, y, X_test, y_test):
    criterion = ['gini', 'entropy']  # , 'log_loss' ]
    splitter = ['best', 'random']
    max_depth = None  # large non-uniform graphs so leave as none
    min_samples_split = [2, 12, 22]
    min_samples_leaf = [1, 11, 21]
    min_weight_fraction_leaf = 0.0  # used to add weights to particular leaves
    max_features = ['auto', 'sqrt', 'log2']
    random_state = None
    max_leaf_nodes = None
    min_impurity_decrease = [0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0]
    class_weight = None
    ccp_alpha = 0.0  # pruning parameter

    model_preds = []
    best_rec = ()
    pred_best = 0
    curr_score = 0

    for i in criterion:
        for j in splitter:
            for k in min_samples_split:
                for l in min_samples_leaf:
                    for m in max_features:
                        for n in min_impurity_decrease:
                            model = sk.tree.DecisionTreeClassifier(criterion=i,
                                                                   splitter=j,
                                                                   min_samples_split=k,
                                                                   min_samples_leaf=l,
                                                                   max_features=m,
                                                                   min_impurity_decrease=n)
                            model.fit(X, y)
                            curr_score = model.score(X_test, y_test)
                            model_preds.append((curr_score, i, j, k, l, m, n))
                            if curr_score > pred_best:
                                best_rec = (curr_score, i, j, k, l, m, n)
                                pred_best = curr_score
    #print(best_rec)
    return model1, best_rec


def generate_random_forest(X, y, X_test, y_test):
    n_estimators = [50, 100, 150, 200, 250]
    criterion = ['gini', 'entropy']
    max_depth = None
    min_samples_split = [2, 12]
    min_samples_leaf = [1, 11]
    min_weight_fraction_leaf = 0.0
    max_features = ['sqrt', 'log2', 'auto']
    max_leaf_nodes = None
    min_impurity_decrease = [0.0, 1.0, 2.0, 3.0, 5.0, 10.0]
    bootstrap = [True, False]
    oob_score = [True, False]  # for use with bootrap = true
    n_jobs = None  # parallel reqs
    random_state = None  #
    verbose = 0
    warm_start = False
    class_weight = None
    ccp_alpha = 0.0  # parameter used for minimal cost-complexity pruning
    max_samples = None

    model_preds = []
    best_rec = ()
    pred_best = 0
    curr_score = 0

    for i in n_estimators:
        for j in criterion:
            for k in min_samples_split:
                for l in min_samples_leaf:
                    for m in max_features:
                        for n in min_impurity_decrease:
                            for o in bootstrap:
                                if o == True:
                                    for p in oob_score:
                                        model = sk.ensemble.RandomForestClassifier(
                                            n_estimators=i,
                                            criterion=j,
                                            min_samples_split=k,
                                            min_samples_leaf=l,
                                            max_features=m,
                                            min_impurity_decrease=n,
                                            bootstrap=o,
                                            oob_score=p).fit(X, y)

                                        curr_score = model.score(X_test, y_test)
                                        model_preds.append((curr_score, i, j, k, l, m, n, o, p))
                                        if curr_score > pred_best:
                                            pred_best = curr_score
                                            best_rec = (curr_score, i, j, k, l, m, n, o, p)
                                else:
                                    model = sk.ensemble.RandomForestClassifier(
                                        n_estimators=i,
                                        criterion=j,
                                        min_samples_split=k,
                                        min_samples_leaf=l,
                                        max_features=m,
                                        min_impurity_decrease=n,
                                        bootstrap=o,
                                        oob_score=p).fit(X, y)

                                    curr_score = model.score(X_test, y_test)
                                    model_preds.append((curr_score, i, j, k, l, m, n, o))
                                    if curr_score > pred_best:
                                        pred_best = curr_score
                                        best_rec = (curr_score, i, j, k, l, m, n, o)

    #print(best_rec)
    return model1, best_rec


def generate_adaboost(X, y, X_test, y_test):
    base_estimator = None  # change this to an array with the rest of the implemented classifiers
    n_estimators = [50, 100, 150, 200]  # base_estimator 's default classifier is DecisionTreeClassifier
    learning_rate = [.3, .5, 1.0, 3.0, 10.0, 100.0]
    algorithm = ['SAMME', 'SAMME.R']
    random_state = None  # only needes if base_estimator has random_state turned on

    model_preds = []
    best_rec = ()
    pred_best = 0
    curr_score = 0

    for j in n_estimators:
        for k in learning_rate:
            for l in algorithm:
                model = sk.ensemble.AdaBoostClassifier(n_estimators=j,
                                                       learning_rate=k,
                                                       algorithm=l).fit(X, y)
                curr_score = model.score(X_test, y_test)
                model_preds.append((curr_score, j, k))
                if curr_score > pred_best:
                    pred_best = curr_score
                    best_rec = (curr_score, j, k)
    #print(best_rec)
    return model1, best_rec


# note that this returned .2275 on initial trial run
def generate_gaussian_naive_bayes(X, y, X_test, y_test):
    var_smoothing = [1 * 10**(-1), 1 * 10**(-3), 1 * 10**(-9), 1*10**(-11), 1 * 10**(-13)]

    model_preds = []
    best_rec = ()
    pred_best = 0
    curr_score = 0

    for i in var_smoothing:
        model = sk.naive_bayes.GaussianNB(var_smoothing=i).fit(X, y)
        curr_score = model.score(X_test, y_test)
        model_preds.append((curr_score, i))
        if curr_score > pred_best:
            best_rec = (curr_score, i)
            pred_best = curr_score

    #print(best_rec)
    return model1, best_rec


def generate_multinomial_naive_bayes(X, y, X_test, y_test):
    alpha = [0, .5, 1, 1.5, 5, 10]  # no smoothing ensured when this is 0 and
    force_alpha = [True, False]  # force_alpha is set to True
    fit_prior = [True, False]
    class_prior = None  # an array of prior probabilities

    model_preds = []
    best_rec = ()
    pred_best = 0
    curr_score = 0

    for i in alpha:
        for j in force_alpha:
            # for k in fit_prior:
            model = sk.naive_bayes.MultinomialNB(alpha=i
                                                 # force_alpha = j
                                                 ).fit(X, y)
            curr_score = model.score(X_test, y_test)
            model_preds.append((curr_score, i))
            if curr_score > pred_best:
                pred_best = curr_score
                best_rec = (curr_score, i)
    #print(best_rec)
    return model1, best_rec


def generate_complement_naive_bayes(X, y, X_test, y_test):
    alpha = [0, .5, 1, 1.5, 5, 10]  # no smoothing ensured when this is 0 and
    force_alpha = [True, False]  # force_alpha is set to True
    norm = [True, False]

    model_preds = []
    best_rec = ()
    pred_best = 0
    curr_score = 0

    for i in alpha:
        for j in norm:
            model = sk.naive_bayes.ComplementNB(alpha=i,
                                                norm=j).fit(X, y)
            curr_score = model.score(X_test, y_test)
            model_preds.append((curr_score, i, j))
            if curr_score > pred_best:
                pred_best = curr_score
                best_rec = (curr_score, i, j)

    #print(best_rec)
    return model1, best_rec


# not working atm
def generate_categorical_naive_bayes(X, y, X_test, y_test):
    # alpha = [ 0, .5, 1, 1.5, 5 ] #no smoothing ensured when this is 0 and
    # force_alpha = [ True, False ]
    # min_categories= [ None  ]
    model_preds = []
    best_rec = ()
    pred_best = 0
    curr_score = 0

    print(len(X), len(X_test))
    print(len(y), len(y_test))
    '''
    for i in alpha:
        for j in min_categories:
            model = sk.naive_bayes.CategoricalNB(alpha=i, min_categories = j).fit(X,y)
            curr_score = model.score(X_test, y_test)
            model_preds.append((curr_score, i, j))
            if curr_score > pred_best:
                pred_best = curr_score
                best_rec = (curr_score, i, j)    
    '''
    best_rec = sk.naive_bayes.CategoricalNB().fit(X, y).score(X_test, y_test)
    #print(best_rec)
    return model1, best_rec


def generate_qda(X, y, X_test, y_test):
    reg_param = [0.0, .1, .2, .3, .4, .5, .6, .7, .8, .9]

    model_preds = []
    best_rec = ()
    pred_best = 0
    curr_score = 0

    for i in reg_param:
        model = sk.discriminant_analysis.QuadraticDiscriminantAnalysis(reg_param=i).fit(X, y)
        curr_score = model.score(X_test, y_test)
        model_preds.append((curr_score, i))
        if curr_score > pred_best:
            pred_best = curr_score
            best_rec = (curr_score, i)

    #print(best_rec)
    return model1, best_rec


def generate_lda(X, y, X_test, y_test):
    solver = ['svd', 'lsqr', 'eigen']
    shrinkage = [0.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 'auto']  # only for use with lsqr and eigen
    tol = [1 * 10**-1, 1 * 10**-2, 1 * 10**-3, 1 * 10**-4, 1 * 10**-5, 1 * 10**-6, 1 * 10**-7, 1 * 10**-8,
           1 * 10**-9]
    # tol is only for use with svd
    # estimate_covariance: look into this parameter

    model_preds = []
    best_rec = ()
    pred_best = 0
    curr_score = 0

    for i in solver:
        if i == 'svd':
            for j in tol:
                model = sk.discriminant_analysis.LinearDiscriminantAnalysis(solver=i,
                                                                            tol=j).fit(X, y)
                curr_score = model.score(X_test, y_test)
                model_preds.append((curr_score, i, j))
                if curr_score > pred_best:
                    pred_best = curr_score
                    best_rec = (curr_score, i, j)
        elif i == 'lsqr' or i == 'eigen':
            for k in shrinkage:
                model = sk.discriminant_analysis.LinearDiscriminantAnalysis(solver=i,
                                                                            shrinkage=k).fit(X, y)
                curr_score = model.score(X_test, y_test)
                model_preds.append((curr_score, i, k))
                if curr_score > pred_best:
                    pred_best = curr_score
                    best_rec = (curr_score, i, k)

    #print(best_rec)
    return model1, best_rec


# not implemented yet
def generate_MLP(X, y, X_test, y_test):
    hidden_layer_sizes= [ (50, ), (100,), (150, ) ]
    activation= [ 'identity', 'logistic', 'tanh', 'relu' ]
    solver= [ 'lbfgs', 'sgd', 'adam' ]
    alpha= [ 0.001 , 0.0001, .00001 ]
    batch_size='auto' # will not be taken into account if solver = lbfgs
    learning_rate= [ 'constant', 'invscaling', 'adaptive' ]
    learning_rate_init=[ .1, 0.001, .00001 ] # only used when solver = 'sgd' or 'adam'
    power_t= [ .3, 0.5, .7 ] # only used when solver='lbfgs', and learning_rate = 'invscaling'
    max_iter=200
    shuffle= [ True, False ] #only used when solver='sgd' or 'adam'
    random_state=None
    tol= [ .001, 0.0001, .00001 ]
    verbose=False
    warm_start=False
    momentum= [ .3, .5, .7, 0.9 ] #only used when solver = 'sgd'
    nesterovs_momentum= [ True, False ]
    early_stopping= [True, False ]
    validation_fraction= [ 0.1 , 0.3 ]
    beta_1= 0.9   #only used when solver='adam'
    beta_2= 0.999 #only used when solver='adam'
    epsilon=1e-08 #only used when solver = 'adam' or 'sgd'
    n_iter_no_change=10
    max_fun=15000 #only used when solver = 'lbfgs'
    model = sk.neural_network.MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                            activation=activation,
                                            solver=solver,
                                            alpha=alpha,
                                            batch_size=batch_size,
                                            learning_rate=learning_rate,
                                            learning_rate_init=learning_rate_init,
                                            power_t=power_t,
                                            max_iter=max_iter,
                                            shuffle=shuffle,
                                            random_state=random_state,
                                            tol=tol,
                                            verbose=verbose,
                                            warm_start=warm_start,
                                            momentum=momentum,
                                            nesterovs_momentum=nesterovs_momentum,
                                            early_stopping=early_stopping,
                                            validation_fraction=validation_fraction,
                                            beta_1=beta_1,
                                            beta_2=beta_2,
                                            epsilon=epsilon,
                                            n_iter_no_change=n_iter_no_change,
                                            max_fun=max_fun)
    return model


def supervised_learning_caller_optimized(alg, X_train, y_train, X_test, y_test):
    if alg == 'knn':
        start_time = time.monotonic()
        model, best_rec = generate_k_nearest_neighbors(X_train, y_train, X_test, y_test)
        training_time = timedelta(seconds=time.monotonic()-start_time)
    elif alg == 'rnn':
        start_time = time.monotonic()
        model, best_rec = generate_radius_nearest_neighbors(X_train, y_train, X_test, y_test)
        training_time = timedelta(seconds=time.monotonic()-start_time)
    elif alg == 'lsvc':
        start_time = time.monotonic()
        model, best_rec = generate_linear_svc(X_train, y_train, X_test, y_test)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'nsvc':
        start_time = time.monotonic()
        model, best_rec = generate_nu_svc(X_train, y_train, X_test, y_test)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'svc':
        start_time = time.monotonic()
        model, best_rec = generate_svc(X_train, y_train, X_test, y_test)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'gp':
        start_time = time.monotonic()
        model, best_rec = generate_gaussian_process(X_train, y_train, X_train, y_train)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'dt':
        start_time = time.monotonic()
        model, best_rec = generate_decision_tree(X_train, y_train, X_test, y_test)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'rf':
        start_time = time.monotonic()
        model, best_rec = generate_random_forest(X_train, y_train, X_test, y_test)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'ada':
        start_time = time.monotonic()
        model, best_rec = generate_adaboost(X_train, y_train, X_test, y_test)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'gnb':
        start_time = time.monotonic()
        model, best_rec = generate_gaussian_naive_bayes(X_train, y_train, X_test, y_test)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'mnb':
        start_time = time.monotonic()
        model, best_rec = generate_multinomial_naive_bayes(X_train, y_train, X_test, y_test)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'compnb':
        start_time = time.monotonic()
        model, best_rec = generate_complement_naive_bayes(X_train, y_train, X_test, y_test)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'catnb':
        start_time = time.monotonic()
        model, best_rec = generate_categorical_naive_bayes(X_train, y_train, X_test, y_test)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'qda':
        start_time = time.monotonic()
        model, best_rec = generate_qda(X_train, y_train, X_test, y_test)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'lda':
        start_time = time.monotonic()
        model, best_rec = generate_lda(X_train, y_train, X_test, y_test)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    return model, best_rec, training_time


def plain_clf_runner(alg, X, y):
    if alg == 'knn':
        start_time = time.monotonic()
        model = sk.neighbors.KNeighborsClassifier().fit(X, y)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'rnn':
        start_time = time.monotonic()
        model = sk.neighbors.RadiusNeighborsClassifier(outlier_label=1, radius=25).fit(X, y)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'lsvc':
        start_time = time.monotonic()
        model = sk.svm.LinearSVC().fit(X, y)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'nsvc':
        start_time = time.monotonic()
        model = sk.svm.NuSVC().fit(X, y)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'svc':
        start_time = time.monotonic()
        model = sk.svm.SVC().fit(X, y)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'gp':
        start_time = time.monotonic()
        model = sk.gaussian_process.GaussianProcessClassifier().fit(X, y)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'dt':
        start_time = time.monotonic()
        model = sk.tree.DecisionTreeClassifier().fit(X, y)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'rf':
        start_time = time.monotonic()
        model = sk.ensemble.RandomForestClassifier().fit(X, y)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'ada':
        start_time = time.monotonic()
        model = sk.ensemble.AdaBoostClassifier().fit(X, y)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'gnb':
        start_time = time.monotonic()
        model = sk.naive_bayes.GaussianNB().fit(X, y)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'mnb':
        start_time = time.monotonic()
        model = sk.naive_bayes.MultinomialNB().fit(X, y)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'compnb':
        start_time = time.monotonic()
        model = sk.naive_bayes.ComplementNB().fit(X, y)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'catnb':
        start_time = time.monotonic()
        model = sk.naive_bayes.CategoricalNB().fit(X, y)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'qda':
        start_time = time.monotonic()
        model = sk.discriminant_analysis.QuadraticDiscriminantAnalysis().fit(X, y)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'lda':
        start_time = time.monotonic()
        model = sk.discriminant_analysis.LinearDiscriminantAnalysis().fit(X, y)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'mlp':
        start_time = time.monotonic()
        model = sk.neural_network.MLPClassifier().fit(X, y)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    return model, training_time


def classifier_name_expand(alg):
    #superv_alg_name = [ 'knn', 'rnn', 'lsvc', 'nsvc', 'svc', 'gp', 'dt', 'rf', 'ada', 'gnb', 'mnb',
    #                    'compnb', 'catnb', 'qda', 'lda']
    if alg ==  'knn':
        alg = 'K-Nearest Neighbors'
    elif alg == 'rnn':
        alg = 'Radius Neighbors'
    elif alg == 'lsvc':
        alg = 'Linear SVC'
    elif alg == 'nsvc':
        alg = 'Nu-SVC'
    elif alg == 'svc':
        alg = 'SVC'
    elif alg == 'gp':
        alg = 'Gaussian Processes'
    elif alg == 'dt':
        alg = 'Decision Tree'
    elif alg == 'rf':
        alg = 'Random Forest'
    elif alg == 'ada':
        alg = 'AdaBoost'
    elif alg == 'gnb':
        alg = 'Gaussian Naive Bayes'
    elif alg == 'mnb':
        alg = 'Multinomial Naive Bayes'
    elif alg == 'compnb':
        alg = 'Complement Naive Bayes'
    elif alg == 'catnb':
        alg = 'Categorical Naive Bayes'
    elif alg == 'qda':
        alg = 'Quadratic Discriminant Analysis'
    elif alg == 'lda':
        alg = 'Linear Discriminant Analysis'
    elif alg == 'mlp':
        alg = 'Multilayer Perceptron'
    return alg

def supervised_methods_evaluation(alg, model, X_test, y_test, ndim,
                                  X_train_size, y_train_size, output_path, training_time, optimized=False, optimizing_tuple=None):
    start_time = time.monotonic()
    y_pred = model.predict(X_test)
    prediction_time = timedelta(seconds=time.monotonic() - start_time)
    output_file_path = output_path + "supervised_methods_evaluation_results/"

    #for dummy testing
    #output_file_path = output_path + "supervised_methods_evaluation_results/dummy_"

    if optimized:
        output_file_path = output_file_path + alg + "_" + str(ndim) + "-dims_optimized_" + "results.txt"
        #output_file_path = output_file_path + "dummy_" + alg + "_" + str(ndim) + "-dims_optimized_" + "results.txt"
        #print(output_file_path)
    else:
        output_file_path = output_file_path + alg + "_" + str(ndim) + "-dims_" + "results.txt"
        #output_file_path = output_file_path + "dummy_" + alg + "_" + str(ndim) + "-dims_" + "results.txt"
        #print("\n" + output_file_path)

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    output_file = open(output_file_path, "w")

    #print("\nClassifier: ", alg)
    output_file.write("Classifier Name: " + classifier_name_expand(alg))

    if optimized:
        output_file.write("\nOptimizing Tuple: " + str(optimizing_tuple))

    output_file.write("\nVector Input Dimensions: " + str(ndim))

    output_file.write("\nTraining Time: " + str(training_time))

    output_file.write("\nPrediction Time: " + str(prediction_time))

    accScore = sk.metrics.accuracy_score(y_test, y_pred)
    output_file.write("\nAccuracy Score: " + str(accScore))
    #print("Accuracy Score: ", accScore)

    skScore = model.score(X_test, y_test)
    output_file.write("\nModel Score: " + str(skScore))
    
    cr = sk.metrics.classification_report(y_test, y_pred)
    output_file.write("\nClassification Report: \n" + str(cr))
    #print("Classification Report: \n", cr)

    cm = sk.metrics.confusion_matrix(y_test, y_pred)
    output_file.write("\nConfusion Matrix: \n" + str(cm))
    #print("Confusion Matrix: \n", cm)
    #cmdisp = sk.metrics.ConfusionMatrixDisplay(cm)
    #cmdisp.plot()
    #plt.show(block=False)

    mcc = sk.metrics.matthews_corrcoef(y_test, y_pred)
    #print("Matthews Correlation Coefficient: ", mcc)
    output_file.write("\nMatthews Correlation Coefficient: " + str(mcc))

    ck = sk.metrics.cohen_kappa_score(y_test, y_pred)
    #print("Cohen's kappa: ", ck)
    output_file.write("\nCohen's kappa: " + str(ck))

    jaccard = sk.metrics.jaccard_score(y_test, y_pred)
    #print("Jaccard Score: ", jaccard)
    output_file.write("\nJaccard Score: " + str(jaccard))

    hingel = sk.metrics.hinge_loss(y_test, y_pred)
    #print("Hinge Loss: ", hingel)
    output_file.write("\nHinge Loss: " + str(hingel))

    hammingl = sk.metrics.hamming_loss(y_test, y_pred)
    #print("Hamming Loss: ", hammingl)
    output_file.write("\nHamming Loss: " + str(hammingl))

    z1l = sk.metrics.zero_one_loss(y_test, y_pred)
    #print("Zero-one Loss: ", z1l)
    output_file.write("\nZero-one Loss: " + str(z1l))

    return output_file_path

# This currently does nothing and I have only been using it for experiment with in-built visualizations
def general_plotter(X, y, ndim, model):
    #graphs, fig = TSNE_2D_plot(X, y, len(y), ndim, return_plot=True)
    #fig_name = model_path + '/' + 'val_vector_' + str(ndim) + '-dims.png'
    #fig_name = './Dummyfig.png'
    #fig.savefig(fig_name)
    #graphs.clf()
    #print(graphs[:,0])
    #print(graphs[:,1])
    #x1, x2 = np.meshgrid(
    #    np.linspace(0, len(graphs[:, 0])),
    #    np.linspace(0, len(graphs[:, 1]))
    #)
    #bounds.plot()
    #y_pred = np.reshape(y, x1.shape)
    #print( len(x1))
    #print(len(x2))
    #ypred = np.reshape(y, len(x1))
    #print(ypred.shape)
    #bounds = sk.inspection.DecisionBoundaryDisplay(xx0=x1, xx1=x2, response = ypred)
    #bounds.plot()
    #prerec = sk.metrics.PredictionErrorDisplay.from_predictions(y, model.predict(X))
    #prerec.plot()
    #plt.plot(x1, x2)
    #plt.scatter(graphs[:,0], graphs[:,1], y )

    #plt.show()

    return 0