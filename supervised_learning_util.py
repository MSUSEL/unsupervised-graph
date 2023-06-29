import numpy as np
import scipy.spatial.distance
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
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from scipy import spatial
import os
from g2v_util import TSNE_2D_plot
import time
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import pickle
import warnings
warnings.filterwarnings("ignore")

X_tr = ""
y_tr = ""
X_te = ""
y_te = "" 

def generate_k_nearest_neighbours(X, y, X_train, y_train, pre_optimized = False):
       
    best_rec = ()
    pred_best = 0
    curr_score = 0
    x_dim, y_dim  = X.shape 
    
    if pre_optimized:
        if y_dim == 2:
            model = sk.neighbors.KNeighborsClassifier(n_neighbors=38,
                                                      weights='distance',
                                                      metric='cityblock',
                                                      n_jobs=-1)
            best_rec = (38, 'distance', 'cityblock')
        elif y_dim == 4:
            model = sk.neighbors.KNeighborsClassifier(n_neighbors=30,
                                                      weights=squared_inverse,
                                                      metric=scipy.spatial.distance.mahalanobis,
                                                      metric_params={'VI': np.linalg.inv(np.cov(X_tr.T))},
                                                      n_jobs=-1)
            best_rec = (30, "squared_inverse", "scipy.spatial.distance.mahalanobis")
        elif y_dim == 8:
            model = sk.neighbors.KNeighborsClassifier(n_neighbors=5,
                                                      weights=squared_inverse,
                                                      metric='cityblock',
                                                      n_jobs=-1)
            best_rec = (5, "squared_inverse", "cityblock")
        elif y_dim == 16:
            model = sk.neighbors.KNeighborsClassifier(n_neighbors=12,
                                                      weights=squared_inverse,
                                                      metric=scipy.spatial.distance.sqeuclidean,
                                                      n_jobs=-1)
            best_rec = (12, 'squared_inverse', 'scipy.spatial.distance.sqeuclidean')
        elif y_dim == 32:
            model = sk.neighbors.KNeighborsClassifier(n_neighbors=10,
                                                      weights=squared_inverse,
                                                      metric=scipy.spatial.distance.sqeuclidean,
                                                      n_jobs=-1)
            best_rec = (10, 'squared_inverse', 'scipy.spatial.distance.sqeuclidean')
        elif y_dim == 64:
            model = sk.neighbors.KNeighborsClassifier(n_neighbors=8,
                                                      weights=squared_inverse,
                                                      metric=scipy.spatial.distance.correlation,
                                                      n_jobs=-1)
            best_rec = (8, 'squared_inverse', 'scipy.spatial.distance.correlation')
        elif y_dim == 128:
            model = sk.neighbors.KNeighborsClassifier(n_neighbors=8,
                                                      weights='distance',
                                                      metric=scipy.spatial.distance.canberra,
                                                      n_jobs=-1)
            best_rec = (8, 'distance', 'scipy.spatial.distance.canberra')

        elif y_dim == 256:
            model = sk.neighbors.KNeighborsClassifier(n_neighbors=6,
                                                      weights=squared_inverse,
                                                      metric=scipy.spatial.distance.correlation,
                                                      n_jobs=-1)
            best_rec = (6, 'squared_inverse', 'scipy.spatial.distance.correlation')
         
        return model, best_rec
    
    metric = [ scipy.spatial.distance.mahalanobis, scipy.spatial.distance.canberra,
               scipy.spatial.distance.chebyshev, scipy.spatial.distance.correlation,
               scipy.spatial.distance.sqeuclidean, 
               'cityblock', 'cosine', 'euclidean']
            
    fin_fut_objs = []
    exe = ThreadPoolExecutor(8)
    for n in metric:
        fin_fut_objs.append(exe.submit(knn_par_metric_helper, n))
    
    for obj in range(len(fin_fut_objs)):
        fin_fut_objs[obj] = fin_fut_objs[obj].result()
    fin_fut_objs = np.array(fin_fut_objs)
    #clf_acc_high = max(fin_fut_objs[:,1])
    clf_acc_high_idx = np.argmax(fin_fut_objs[:,1])
    model1 = fin_fut_objs[clf_acc_high_idx][0]
    best_rec = fin_fut_objs[clf_acc_high_idx][2]
    return model1, best_rec

def knn_par_metric_helper(metric_passed): 
    print(metric_passed)
    #need to add cross-validation here
    num_neigh = list(range(5,9))
    
    weights = ['distance', inverse_weights, squared_inverse]
    pred_best = 0
    model1 = ""
    best_rec = []
    for j in num_neigh:
        for m in weights:
            print(metric_passed, j, m)
            if metric_passed == scipy.spatial.distance.mahalanobis:
                model = sk.neighbors.KNeighborsClassifier(n_neighbors=j,
                                                          weights=m,
                                                          metric=metric_passed,
                                                          metric_params={'VI': np.linalg.inv(np.cov(X_tr.T))},
                                                          n_jobs=-1).fit(X_tr,y_tr)
                y_pred = model.predict(X_te)
                curr_score = sk.metrics.accuracy_score(y_te, y_pred)
            else:
                model = sk.neighbors.KNeighborsClassifier(n_neighbors=j,
                                                          weights=m,
                                                          metric=metric_passed,
                                                          n_jobs=-1).fit(X_tr,y_tr)
                y_pred = model.predict(X_te)
                curr_score = sk.metrics.accuracy_score(y_te, y_pred)
            if curr_score > pred_best:
                best_rec = (curr_score, metric_passed, j, m)
                model1 = model
                pred_best = curr_score 
                print(model1, " ", pred_best, " ", best_rec)
    return model1, pred_best, best_rec
    
def knn_par_neigh_helper(num_neigh):
    print(num_neigh)
    weights = ['distance',
               inverse_weights, squared_inverse]
    metric = [ scipy.spatial.distance.mahalanobis,
               scipy.spatial.distance.chebyshev, scipy.spatial.distance.correlation,
               scipy.spatial.distance.sqeuclidean, 
               'cityblock', 'cosine', 'euclidean']
    pred_best = 0
    model1 = ""
    best_rec = []
    for j in weights:
        for m in metric:
            print(num_neigh, j, m)
            if m == scipy.spatial.distance.mahalanobis:
                model = sk.neighbors.KNeighborsClassifier(n_neighbors=num_neigh,
                                                          weights=j,
                                                          metric=m,
                                                          metric_params={'VI': np.linalg.inv(np.cov(X_tr.T))},
                                                          n_jobs=-1).fit(X_tr,y_tr)
                y_pred = model.predict(X_te)
                curr_score = sk.metrics.accuracy_score(y_te, y_pred)
            else:
                model = sk.neighbors.KNeighborsClassifier(n_neighbors=num_neigh,
                                                          weights=j,
                                                          metric=m,
                                                          n_jobs=-1).fit(X_tr,y_tr)
                y_pred = model.predict(X_te)
                curr_score = sk.metrics.accuracy_score(y_te, y_pred)
            if curr_score > pred_best:
                best_rec = (curr_score, num_neigh, j, m)
                model1 = model
                pred_best = curr_score 
    #print("Completed: ", model1, " ", pred_best, " ", best_rec)
    return model1, pred_best, best_rec

def inverse_weights(inpVec):
    return 1/inpVec

def squared_inverse(inpVec):
    return 1/(inpVec)**2


def generate_svc(X, y, X_test, y_test, pre_optimized=False):
    x_dim, y_dim  = X.shape 
    
    if pre_optimized:
        if y_dim == 2:
            model = sk.svm.SVC(C=10,
                               kernel='rbf',
                               gamma='scale',
                               tol=.01,
                               random_state = 25,
                               probability=True,
                               max_iter = -1)
            best_rec = ('kernel=RBF',  'C=10', 'gamma=scale', 'tol=.01') 
        elif y_dim == 4:
            model = sk.svm.SVC(C=10,
                               kernel='poly',
                               degree=5,
                               gamma='auto',
                               coef0=1.0,
                               tol=.01,
                               random_state = 25,
                               probability=True,
                               max_iter = -1)
            best_rec = ('kernel=poly',  'C=10', 'degree=5', 'gamma=auto', 'coef0=1.0', 'tol=.01') 
        elif y_dim == 8:
            model = sk.svm.SVC(C=10,
                               kernel='poly',
                               degree=5,
                               gamma='scale',
                               coef0=1.0,
                               tol=.001,
                               random_state = 25,
                               probability=True,
                               max_iter = -1)
            best_rec = ('kernel=poly',  'C=10', 'degree=5', 'gamma=scale', 'coef0=1.0', 'tol=.001') 
        elif y_dim == 16:
            model = sk.svm.SVC(C=10,
                               kernel='rbf',
                               gamma='auto',
                               tol=.001,
                               random_state = 25,
                               probability=True,
                               max_iter = -1)
            best_rec = ('kernel=RBF',  'C=10', 'gamma=auto', 'tol=.001')            
        elif y_dim == 32:
            model = sk.svm.SVC(C=10,
                               kernel='rbf',
                               gamma='scale',
                               tol=.01,
                               random_state = 25,
                               probability=True,
                               max_iter = -1)
            best_rec = ('kernel=RBF',  'C=10', 'gamma=scale', 'tol=.01') 
        elif y_dim == 64:
            model = sk.svm.SVC(C=10,
                               kernel='rbf',
                               gamma='auto',
                               tol=.01,
                               random_state = 25,
                               probability=True,
                               max_iter = -1)
            best_rec = ('kernel=RBF',  'C=10', 'gamma=auto', 'tol=.01') 
        elif y_dim == 128:
            model = sk.svm.SVC(C=10,
                               kernel='rbf',
                               gamma='scale',
                               tol=.001,
                               random_state = 25,
                               probability=True,
                               max_iter = -1)
            best_rec = ('kernel=RBF',  'C=10', 'gamma=scale', 'tol=.001')  
        elif y_dim == 256:
            model = sk.svm.SVC(C=10,
                               kernel='rbf',
                               gamma='auto',
                               tol=.01,
                               random_state = 25,
                               probability=True,
                               max_iter = -1)
            best_rec = ('kernel=RBF',  'C=10', 'gamma=auto', 'tol=.01') 
        return model, best_rec
            
    kernel = ['linear', 'rbf', 'sigmoid', 'poly']
    fin_fut_objs = []
    exe = ThreadPoolExecutor(8)
    for n in kernel:
        fin_fut_objs.append(exe.submit(svc_par_helper, n))
    
    for obj in range(len(fin_fut_objs)):
        fin_fut_objs[obj] = fin_fut_objs[obj].result()
    fin_fut_objs = np.array(fin_fut_objs)
    #clf_acc_high = max(fin_fut_objs[:,1])
    clf_acc_high_idx = np.argmax(fin_fut_objs[:,1])
    model1 = fin_fut_objs[clf_acc_high_idx][0]
    best_rec = fin_fut_objs[clf_acc_high_idx][2]
         
    #print(best_rec)
    return model1, best_rec

def svc_par_helper(passed_kernel):
    model_preds = []
    best_rec = ()
    pred_best = 0
    curr_score = 0
    gamma = [ 'scale', 'auto']
    C = [.1, 1, 10]
    coef0 = [0.0, .5, 1.0]
    tol = [.01, 0.001, .0001]
    if passed_kernel == 'linear':
        for i in C:
            for j in tol:
                print((passed_kernel, i, j))
                model = sk.svm.SVC(C=i,
                                   kernel=passed_kernel,
                                   tol=j,
                                   random_state = 25,
                                   probability=True,
                                   max_iter = -1)
                model_scores = cross_val_score(model, X_tr, y_tr, cv = 5, n_jobs = -1 )
                curr_score = model_scores.mean()
                model_preds.append((curr_score, passed_kernel, i, j))
                print((curr_score, passed_kernel, i, j))
                if curr_score > pred_best:
                    model1 = model
                    pred_best = curr_score
                    best_rec = (curr_score, passed_kernel, i, j) 
    elif passed_kernel == 'rbf':
        for i in C:
            for j in gamma:
                for k in tol:
                    print((passed_kernel, i, j, k))
                    model = sk.svm.SVC(C=i,
                                       kernel=passed_kernel,
                                       gamma=j,
                                       tol=k,
                                       random_state = 25,
                                       probability=True,
                                       max_iter = -1)
                    model_scores = cross_val_score(model, X_tr, y_tr, cv = 5, n_jobs = -1 )
                    curr_score = model_scores.mean()
                    model_preds.append((curr_score, passed_kernel, i, j, k))
                    print((curr_score, passed_kernel, i, j, k))
                    if curr_score > pred_best:
                        model1 = model
                        pred_best = curr_score
                        best_rec = (curr_score, passed_kernel, i, j, k) 
    elif passed_kernel == 'sigmoid':
        for i in C:
            for j in gamma:
                for k in coef0:
                    for l in tol:
                        print((passed_kernel, i, j, k, l))
                        model = sk.svm.SVC(C=i,
                                           kernel=passed_kernel,
                                           gamma=j,
                                           coef0=k,
                                           tol=l,
                                           random_state = 25,
                                           probability=True,
                                           max_iter = -1)
                        model_scores = cross_val_score(model, X_tr, y_tr, cv = 5, n_jobs = -1 )
                        curr_score = model_scores.mean()
                        model_preds.append((curr_score, passed_kernel, i, j, k, l))
                        print((curr_score, passed_kernel, i, j, k, l))
                        if curr_score > pred_best:
                            model1 = model
                            pred_best = curr_score
                            best_rec = (curr_score, passed_kernel, i, j, k, l) 
    elif passed_kernel == 'poly':
        degree = [1,2,3,5]
        for i in degree:
            for j in C:
                for k in gamma:
                    for l in coef0:
                        for m in tol:
                            print(passed_kernel, i, j, k, l, m)
                            model = sk.svm.SVC(C=j,
                                               kernel=passed_kernel,
                                               degree=i,
                                               gamma=k,
                                               coef0=l,
                                               tol=m,
                                               random_state = 25,
                                               probability=True,
                                               max_iter = -1)
                            model_scores = cross_val_score(model, X_tr, y_tr, cv = 5, n_jobs = -1 )
                            curr_score = model_scores.mean()
                            model_preds.append((curr_score, passed_kernel, j, i, k, l, m))
                            print((curr_score, passed_kernel, j, i, k, l, m))
                            if curr_score > pred_best:
                                model1 = model
                                pred_best = curr_score
                                best_rec = (curr_score, passed_kernel, j, i, k, l, m) 
    return model1, pred_best, best_rec


def generate_gaussian_process(X, y, X_test, y_test, pre_optimized = False):
    x_dim, y_dim = X.shape
    if pre_optimized:
        if y_dim == 2:
            kernel_passed = sk.gaussian_process.kernels.RationalQuadratic(alpha=2.5, length_scale=.5)
            model1 = sk.gaussian_process.GaussianProcessClassifier(  kernel = kernel_passed,
                                                                    optimizer='fmin_l_bfgs_b',
                                                                    n_restarts_optimizer=0,
                                                                    warm_start=True,
                                                                    max_iter_predict=100,
                                                                    random_state=25,
                                                                    n_jobs=-1) 
            best_rec = ('RationalQuadratic', 'alpha=2.5', 'length_scale=.5')
        elif y_dim == 4:
            kernel_passed = sk.gaussian_process.kernels.RationalQuadratic(alpha=2.5, length_scale=1)
            model1 = sk.gaussian_process.GaussianProcessClassifier(  kernel = kernel_passed,
                                                                    optimizer='fmin_l_bfgs_b',
                                                                    n_restarts_optimizer=0,
                                                                    warm_start=True,
                                                                    max_iter_predict=100,
                                                                    random_state=25,
                                                                    n_jobs=-1) 
            best_rec = ('RationalQuadratic', 'alpha=2.5', 'length_scale=1')
        elif y_dim == 8:
            kernel_passed = sk.gaussian_process.kernels.PairwiseKernel(gamma=.8, metric='poly')
            model1 = sk.gaussian_process.GaussianProcessClassifier( kernel = kernel_passed,
                                                                    optimizer='fmin_l_bfgs_b',
                                                                    n_restarts_optimizer=0,
                                                                    warm_start=True,
                                                                    max_iter_predict=100,
                                                                    random_state=25,
                                                                    n_jobs=-1) 
            best_rec = ('PairwiseKernel', 'metric=poly', 'gamma=.8')
        elif y_dim == 16:
            kernel_passed = sk.gaussian_process.kernels.PairwiseKernel(metric='poly', gamma=.653)
            model1 = sk.gaussian_process.GaussianProcessClassifier(  kernel = kernel_passed,
                                                                     optimizer='fmin_l_bfgs_b',
                                                                     n_restarts_optimizer=0,
                                                                     warm_start=True,
                                                                     max_iter_predict=100,
                                                                     random_state=25,
                                                                     n_jobs=-1) 
            best_rec = ('PairwiseKernel', 'metric=poly', 'gamma=.8')
        elif y_dim == 32:
            kernel_passed = sk.gaussian_process.kernels.PairwiseKernel(metric='polynomial', gamma=.6538)
            model1 = sk.gaussian_process.GaussianProcessClassifier(  kernel = kernel_passed,
                                                                     optimizer='fmin_l_bfgs_b',
                                                                     n_restarts_optimizer=0,
                                                                     warm_start=True,
                                                                     max_iter_predict=100,
                                                                     random_state=25,
                                                                     n_jobs=-1) 
            best_rec = ('PairwiseKernel', 'metric=polynomial', 'gamma=.6538')
        elif y_dim == 64:
            kernel_passed = sk.gaussian_process.kernels.PairwiseKernel(metric='polynomial', gamma=.5)
            model1 = sk.gaussian_process.GaussianProcessClassifier(  kernel = kernel_passed,
                                                                     optimizer='fmin_l_bfgs_b',
                                                                     n_restarts_optimizer=0,
                                                                     warm_start=True,
                                                                     max_iter_predict=100,
                                                                     random_state=25,
                                                                     n_jobs=-1) 
            best_rec = ('PairwiseKernel', 'metric=polynomial', 'gamma=.5')
        elif y_dim == 128:
            kernel_passed = sk.gaussian_process.kernels.PairwiseKernel(metric='polynomial', gamma=.5)
            model1 = sk.gaussian_process.GaussianProcessClassifier(  kernel = kernel_passed,
                                                                     optimizer='fmin_l_bfgs_b',
                                                                     n_restarts_optimizer=0,
                                                                     warm_start=True,
                                                                     max_iter_predict=100,
                                                                     random_state=25,
                                                                     n_jobs=-1) 
            best_rec = ('PairwiseKernel', 'metric=polynomial', 'gamma=.5')
        elif y_dim == 256:
            kernel_passed = sk.gaussian_process.kernels.PairwiseKernel(metric='polynomial', gamma=1.2)
            model1 = sk.gaussian_process.GaussianProcessClassifier(  kernel = kernel_passed,
                                                                     optimizer='fmin_l_bfgs_b',
                                                                     n_restarts_optimizer=0,
                                                                     warm_start=True,
                                                                     max_iter_predict=100,
                                                                     random_state=25,
                                                                     n_jobs=-1) 
            best_rec = ('PairwiseKernel', 'metric=polynomial', 'gamma=1.2')
        
        return model1, best_rec
    
        
    #kernel = ['RBF', 'matern', 'rationalquadratic', 'dotproduct', 'linear', 'poly', 'polynomial', 'laplacian', 'sigmoid', 'cosine' , 'None']  
    
    gamma = [  1.221, 1.222, 1.223, 1.224, 1.225, 1.226, 1.227, 1.228, 1.229 ]    

    #mult = [ 1.0, 2.0, 3.0 , 4.0, 5.0, 6.0, 7.0, 8.0]
    
    fin_fut_objs = []
    exe = ThreadPoolExecutor(8)
    for n in gamma:
        fin_fut_objs.append(exe.submit(gp_poly_helper, n))
    
    for obj in range(len(fin_fut_objs)):
        fin_fut_objs[obj] = fin_fut_objs[obj].result()
    fin_fut_objs = np.array(fin_fut_objs)
    #clf_acc_high = max(fin_fut_objs[:,1])
    clf_acc_high_idx = np.argmax(fin_fut_objs[:,1])
    model1 = fin_fut_objs[clf_acc_high_idx][0]
    best_rec = fin_fut_objs[clf_acc_high_idx][2]
         
    return model1, best_rec

def gp_rbf_helper(mult):
    kernel_passed = mult * sk.gaussian_process.kernels.RBF(1.0) 
    model = sk.gaussian_process.GaussianProcessClassifier(  kernel = kernel_passed,
                                                            optimizer='fmin_l_bfgs_b',
                                                            n_restarts_optimizer=0,
                                                            warm_start=False,
                                                            max_iter_predict=100,
                                                            random_state=25,
                                                            n_jobs=-1) 
    print("Fitting model: ", kernel_passed)
    model_scores = cross_val_score(model, X_tr, y_tr, cv = 5, n_jobs = -1 )
    curr_score = model_scores.mean()
    print(kernel_passed, curr_score, gamma)
    best_rec = (kernel_passed, curr_score, gamma)
    
    return model, curr_score, best_rec

def gp_poly_helper(gamma):
    kernel_passed = sk.gaussian_process.kernels.PairwiseKernel(gamma=gamma, metric='poly')
    model = sk.gaussian_process.GaussianProcessClassifier(  kernel = kernel_passed,
                                                            optimizer='fmin_l_bfgs_b',
                                                            n_restarts_optimizer=0,
                                                            warm_start=True,
                                                            max_iter_predict=100,
                                                            random_state=25,
                                                            n_jobs=-1) 
    print("Fitting model: ", kernel_passed)
    model_scores = cross_val_score(model, X_tr, y_tr, cv = 5, n_jobs = -1 )
    curr_score = model_scores.mean()
    print(kernel_passed, curr_score, gamma)
    best_rec = (kernel_passed, curr_score, gamma)
    return model, curr_score, best_rec

def gp_par_helper(kernel_passed):
    model1 = ""
    best_rec = ()
    pred_best = 0
    curr_score = 0
    
    if kernel_passed == 'RBF':
        length_scale = [ 1.4 ]
        for i in length_scale:
            kernel_passed = sk.gaussian_process.kernels.RBF(length_scale = i)
            model = sk.gaussian_process.GaussianProcessClassifier(  kernel = kernel_passed,
                                                                    optimizer='fmin_l_bfgs_b',
                                                                    n_restarts_optimizer=0,
                                                                    warm_start=True,
                                                                    max_iter_predict=100,
                                                                    random_state=25,
                                                                    n_jobs=-1) 
            print("Fitting model: ", kernel_passed)
            model_scores = cross_val_score(model, X_tr, y_tr, cv = 5, n_jobs = -1 )
            curr_score = model_scores.mean()
            if curr_score > pred_best:
                best_rec = (curr_score, kernel_passed, i)
                model1 = model
                pred_best = curr_score 
            print(kernel_passed, curr_score, i)
    elif kernel_passed == 'matern': 
        length_scale = [ .4]
        nu = [1.1,1.2,1.3,1.4,1.5]
        for i in length_scale:
            for j in nu:
                kernel_passed = sk.gaussian_process.kernels.Matern(length_scale = i, nu = j)
                model = sk.gaussian_process.GaussianProcessClassifier(  kernel = kernel_passed,
                                                                        optimizer='fmin_l_bfgs_b',
                                                                        n_restarts_optimizer=0,
                                                                        warm_start=True,
                                                                        max_iter_predict=100,
                                                                        random_state=25,
                                                                        n_jobs=-1) 
            print("Fitting model: ", kernel_passed)
            model_scores = cross_val_score(model, X_tr, y_tr, cv = 5, n_jobs = -1 )
            curr_score = model_scores.mean()
            if curr_score > pred_best:
                best_rec = (curr_score, kernel_passed, i, j)
                model1 = model
                pred_best = curr_score 
            print(kernel_passed, curr_score, i, j)
                
    elif kernel_passed == 'rationalquadratic': 
        length_scale = [ .5]
        alpha = [5,5.1,5.2,5.3,5.4,5.5]
        for i in length_scale:
            for j in alpha:
                kernel_passed = sk.gaussian_process.kernels.RationalQuadratic(length_scale = i, alpha = j)
                model = sk.gaussian_process.GaussianProcessClassifier(  kernel = kernel_passed,
                                                                        optimizer='fmin_l_bfgs_b',
                                                                        n_restarts_optimizer=0,
                                                                        warm_start=True,
                                                                        max_iter_predict=100,
                                                                        random_state=25,
                                                                        n_jobs=-1) 
            print("Fitting model: ", kernel_passed)
            model_scores = cross_val_score(model, X_tr, y_tr, cv = 5, n_jobs = -1 )
            curr_score = model_scores.mean()
            if curr_score > pred_best:
                best_rec = (curr_score, kernel_passed, i, j)
                model1 = model
                pred_best = curr_score 
            print(kernel_passed, curr_score, i, j)
    elif kernel_passed == 'None':
        model = sk.gaussian_process.GaussianProcessClassifier(  kernel = None,
                                                                optimizer='fmin_l_bfgs_b',
                                                                n_restarts_optimizer=0,
                                                                warm_start=True,
                                                                max_iter_predict=100,
                                                                random_state=25,
                                                                n_jobs=-1) 
        print("Fitting model: ", kernel_passed)
        model_scores = cross_val_score(model, X_tr, y_tr, cv = 5, n_jobs = -1 )
        curr_score = model_scores.mean()
        if curr_score > pred_best:
            best_rec = (curr_score, kernel_passed)
            model1 = model
            pred_best = curr_score 
        print(kernel_passed, curr_score)
    elif kernel_passed == 'dotproduct': 
        #sigma = [2,4,6,8,10]
        #for i in sigma:
        kernel_passed = sk.gaussian_process.kernels.DotProduct()
        model = sk.gaussian_process.GaussianProcessClassifier(  kernel = kernel_passed,
                                                                    optimizer='fmin_l_bfgs_b',
                                                                    n_restarts_optimizer=0,
                                                                    warm_start=True,
                                                                    max_iter_predict=100,
                                                                    random_state=25,
                                                                    n_jobs=-1) 
        print("Fitting model: ", kernel_passed)
        model_scores = cross_val_score(model, X_tr, y_tr, cv = 5, n_jobs = -1 )
        curr_score = model_scores.mean()
        if curr_score > pred_best:
            best_rec = (curr_score, kernel_passed)
            model1 = model
            pred_best = curr_score 
        print(kernel_passed, curr_score)
    elif kernel_passed == 'linear': 
        #gamma = [.5, 1, 1.5]
        #for i in gamma:
        kernel_passed = sk.gaussian_process.kernels.PairwiseKernel( metric='linear')
        model = sk.gaussian_process.GaussianProcessClassifier(  kernel = kernel_passed,
                                                                    optimizer='fmin_l_bfgs_b',
                                                                    n_restarts_optimizer=0,
                                                                    warm_start=True,
                                                                    max_iter_predict=100,
                                                                    random_state=25,
                                                                    n_jobs=-1) 
        print("Fitting model: ", kernel_passed)
        model_scores = cross_val_score(model, X_tr, y_tr, cv = 5, n_jobs = -1 )
        curr_score = model_scores.mean()
        if curr_score > pred_best:
            best_rec = (curr_score, kernel_passed)
            model1 = model
            pred_best = curr_score 
        print(kernel_passed, curr_score)
    elif kernel_passed == 'poly': 
        gamma = [.1,.2,.3,.4,.5,.6,.7, 1.2, 1.3, 1.4, 1.5, 1.6]
        for i in gamma:
            kernel_passed = sk.gaussian_process.kernels.PairwiseKernel(gamma=i, metric='poly')
            model = sk.gaussian_process.GaussianProcessClassifier(  kernel = kernel_passed,
                                                                    optimizer='fmin_l_bfgs_b',
                                                                    n_restarts_optimizer=0,
                                                                    warm_start=True,
                                                                    max_iter_predict=100,
                                                                    random_state=25,
                                                                    n_jobs=-1) 
            print("Fitting model: ", kernel_passed)
            model_scores = cross_val_score(model, X_tr, y_tr, cv = 5, n_jobs = -1 )
            curr_score = model_scores.mean()
            if curr_score > pred_best:
                best_rec = (curr_score, kernel_passed, i)
                model1 = model
                pred_best = curr_score 
            print(kernel_passed, curr_score, i)
    elif kernel_passed == 'polynomial': 
        gamma = [.7,.8,.9, 1, 1.1, 1.2]
        for i in gamma:
            kernel_passed = sk.gaussian_process.kernels.PairwiseKernel(gamma=i, metric='polynomial')
            model = sk.gaussian_process.GaussianProcessClassifier(  kernel = kernel_passed,
                                                                    optimizer='fmin_l_bfgs_b',
                                                                    n_restarts_optimizer=0,
                                                                    warm_start=True,
                                                                    max_iter_predict=100,
                                                                    random_state=25,
                                                                    n_jobs=-1) 
            print("Fitting model: ", kernel_passed)
            model_scores = cross_val_score(model, X_tr, y_tr, cv = 5, n_jobs = -1 )
            curr_score = model_scores.mean()
            if curr_score > pred_best:
                best_rec = (curr_score, kernel_passed, i)
                model1 = model
                pred_best = curr_score 
            print(kernel_passed, curr_score, i)
    elif kernel_passed == 'laplacian': 
        #gamma = [.5, 1, 1.5]
        #for i in gamma:
        kernel_passed = sk.gaussian_process.kernels.PairwiseKernel(metric='laplacian')
        model = sk.gaussian_process.GaussianProcessClassifier(  kernel = kernel_passed,
                                                                    optimizer='fmin_l_bfgs_b',
                                                                    n_restarts_optimizer=0,
                                                                    warm_start=True,
                                                                    max_iter_predict=100,
                                                                    random_state=25,
                                                                    n_jobs=-1) 
        print("Fitting model: ", kernel_passed)
        model_scores = cross_val_score(model, X_tr, y_tr, cv = 5, n_jobs = -1 )
        curr_score = model_scores.mean()
        if curr_score > pred_best:
            best_rec = (curr_score, kernel_passed)
            model1 = model
            pred_best = curr_score 
        print(kernel_passed, curr_score)
        
    return model1, pred_best, best_rec


def supervised_learning_caller_optimized(alg, X_train, y_train, X_test, y_test):
    global X_tr, y_tr, X_te, y_te
    X_tr = X_train
    y_tr = y_train
    X_te = X_test
    y_te = y_test
    if alg == 'knn':
        start_time = time.monotonic()
        model, best_rec = generate_k_nearest_neighbours(X_train, y_train, X_test, y_test, pre_optimized=True)
        training_time = timedelta(seconds=time.monotonic()-start_time)
    elif alg == 'svc':
        start_time = time.monotonic()
        model, best_rec = generate_svc(X_train, y_train, X_test, y_test, pre_optimized=True)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'gp':
        start_time = time.monotonic()
        model, best_rec = generate_gaussian_process(X_train, y_train, X_test, y_test, pre_optimized=True)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    return model, best_rec, training_time


def plain_clf_runner(alg, X, y):
    if alg == 'knn':
        start_time = time.monotonic()
        model = sk.neighbors.KNeighborsClassifier()
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'rnn':
        start_time = time.monotonic()
        model = sk.neighbors.RadiusNeighborsClassifier(outlier_label=1, radius=25)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'lsvc':
        start_time = time.monotonic()
        model = sk.svm.LinearSVC()
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'nsvc':
        start_time = time.monotonic()
        model = sk.svm.NuSVC()
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'svc':
        start_time = time.monotonic()
        model = sk.svm.SVC(probability=True)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'gp':
        start_time = time.monotonic()
        model = sk.gaussian_process.GaussianProcessClassifier()
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'dt':
        start_time = time.monotonic()
        model = sk.tree.DecisionTreeClassifier()
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'rf':
        start_time = time.monotonic()
        model = sk.ensemble.RandomForestClassifier()
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'ada':
        start_time = time.monotonic()
        model = sk.ensemble.AdaBoostClassifier()
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'gnb':
        start_time = time.monotonic()
        model = sk.naive_bayes.GaussianNB()
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'mnb':
        start_time = time.monotonic()
        model = sk.naive_bayes.MultinomialNB()
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'compnb':
        start_time = time.monotonic()
        model = sk.naive_bayes.ComplementNB()
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'catnb':
        start_time = time.monotonic()
        model = sk.naive_bayes.CategoricalNB()
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'qda':
        start_time = time.monotonic()
        model = sk.discriminant_analysis.QuadraticDiscriminantAnalysis()
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'lda':
        start_time = time.monotonic()
        model = sk.discriminant_analysis.LinearDiscriminantAnalysis()
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'mlp':
        start_time = time.monotonic()
        model = sk.neural_network.MLPClassifier()
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

def supervised_methods_evaluation(alg, model, X, y, X_test, y_test, ndim,
                                  X_train_size, y_train_size, output_path, training_time,optimized=False, optimizing_tuple=None):
    
    start_time = time.monotonic()
    acc_scores = cross_val_score(model, X, y, cv = 5, n_jobs = -1 )
    training_time = timedelta(seconds=time.monotonic() - start_time)
    
    start_time = time.monotonic()
    y_pred = model.fit(X,y).predict(X_test)
    prediction_time = timedelta(seconds=time.monotonic() - start_time)
    
    #output_file_path = output_path + "supervised_methods_evaluation_results/" + alg + "/"
    output_path = output_path + "supervised_methods_evaluation_results/" + alg + "/"
    #for dummy testing
    #output_file_path = output_path + "supervised_methods_evaluation_results/dummy_"

    if optimized:
        output_path = output_path + "/optimized_results/"  
        summary_file_path = output_path + alg + "_summary_optimized.csv"
        output_file_path = output_path + alg + "_" + str(ndim) + "-dims_optimized_" + "results.txt"
        roc_data_path = output_path + alg + "_" + str(ndim) + "-dims_optimized_" + "roc_data.csv"
        #output_file_path = output_file_path + "dummy_" + alg + "_" + str(ndim) + "-dims_optimized_" + "results.txt"
        #print(output_file_path)
    else:
        output_path = output_path + "/plain_results/"
        summary_file_path = output_path + alg + "_summary_plain.csv"
        output_file_path = output_path + alg + "_" + str(ndim) + "-dims_" + "results.txt"
        roc_data_path = output_path + alg + "_" + str(ndim) + "-dims_" + "roc_data.csv"
        
        #output_file_path = outpu-t_file_path + "dummy_" + alg + "_" + str(ndim) + "-dims_" + "results.txt"
        #print("\n" + output_file_path)

    os.makedirs(os.path.dirname(summary_file_path), exist_ok=True)

    output_file = open(output_file_path, "w")
    output_file2 = open(summary_file_path, "a")
    output_file3 = open(roc_data_path, "a")
    
    output_file.write("Classifier Name: " + classifier_name_expand(alg))
    output_file2.write(classifier_name_expand(alg) + ", ")

    if optimized:
        output_file.write("\nOptimizing Tuple: " + str(optimizing_tuple))
        output_file2.write(str(optimizing_tuple) + ", ")

    output_file.write("\nVector Input Dimensions: " + str(ndim))
    output_file2.write(str(ndim) + ", ")

    output_file.write("\nTraining Time: " + str(training_time) + ", ")

    output_file.write("\nPrediction Time: " + str(prediction_time) + ", ")
    
    
    output_file.write("\nAccuracy Scores During Training: " + str(acc_scores))
    output_file.write("\nAccuracy Scores Mean During Training: " + str(acc_scores.mean()))
    output_file.write("\nAccuracy Scores Standard Deviation During Training: " + str(acc_scores.std()))
    
    output_file2.write(str(acc_scores) + ", " + str(acc_scores.mean()) + str(acc_scores.std()) )
    
    accScore = sk.metrics.accuracy_score(y_test, y_pred)
    output_file.write("\nAccuracy Score on Test Set: " + str(accScore))
    
    output_file2.write(str(acc_scores.mean()) + ", ")
    output_file2.write(str(acc_scores.std()) + ", ")
    output_file2.write(str(acc_scores) + ", ")
    #print("Accuracy Score: ", accScore)

    preScore = sk.metrics.precision_score(y_test, y_pred)
    output_file.write("\nPrecision Score: " + str(preScore))
    output_file2.write(str(preScore) + ", ")
    
    recScore = sk.metrics.recall_score(y_test, y_pred)
    output_file.write("\nRecall Score: " + str(recScore))
    output_file2.write(str(recScore) + ", ")                  
                      
    f1Score = sk.metrics.recall_score(y_test, y_pred)
    output_file.write("\nF1 Score: " + str(f1Score))
    output_file2.write(str(f1Score) + ", ")                       

    cm = sk.metrics.confusion_matrix(y_test, y_pred)
    output_file.write("\nConfusion Matrix: \n" + str(cm))
    cmdisp = sk.metrics.ConfusionMatrixDisplay(cm, display_labels=['Benign', 'Malignant'])
    cmdisp.plot()
    plt.title(classifier_name_expand(alg) + " " + str(ndim) + "-Dimensions Confusion Matrix")
    plt.savefig(output_path + alg + "_" + str(ndim) + "-dims_" + "cm.png")
    plt.close()
    #plt.show(block=False)
    
    y_score = model.predict_proba(X_test)
    fig_obj = plt.figure()
    y_score_rav = y_score[:,1]
    ras = sk.metrics.roc_auc_score(y_test, y_score_rav)
    fpr, tpr, thrsh = sk.metrics.roc_curve(y_test, y_score_rav)
    plt.plot(fpr,tpr,label=str(ndim) + "-Dim")
    plt.title(classifier_name_expand(alg) + " " + str(ndim) + "-Dimensions ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    text_box = AnchoredText("ROC AUC Score: " + str(ras), frameon=True, loc=4, pad=0.5)
    plt.setp(text_box.patch, facecolor='white', alpha=0.5)
    plt.gca().add_artist(text_box)
    plt.savefig(output_path + alg + "_" + str(ndim) + "-dims_" + "ras.png")
    plt.close()
    
    output_file.write("\nROC AUC Score: " + str(ras))
    output_file2.write(str(ras)+ ", ")
    output_file3.write("fpr\n")
    output_file3.write(str(fpr))
    output_file3.write("tpr\n")
    output_file3.write(str(tpr))
    output_file3.write("thrsh\n")
    output_file3.write(str(thrsh)) 
    
    mcc = sk.metrics.matthews_corrcoef(y_test, y_pred)
    #print("Matthews Correlation Coefficient: ", mcc)
    output_file.write("\nMatthews Correlation Coefficient: " + str(mcc))
    output_file2.write(str(mcc) + ", ")

    ck = sk.metrics.cohen_kappa_score(y_test, y_pred)
    #print("Cohen's kappa: ", ck)
    output_file.write("\nCohen's kappa: " + str(ck))
    output_file2.write(str(ck) + ", ")

    jaccard = sk.metrics.jaccard_score(y_test, y_pred)
    #print("Jaccard Score: ", jaccard)
    output_file.write("\nJaccard Score: " + str(jaccard))
    output_file2.write(str(jaccard) + ", ")

    hingel = sk.metrics.hinge_loss(y_test, y_pred)
    #print("Hinge Loss: ", hingel)
    output_file.write("\nHinge Loss: " + str(hingel))
    output_file2.write(str(hingel) + ", ")

    hammingl = sk.metrics.hamming_loss(y_test, y_pred)
    #print("Hamming Loss: ", hammingl)
    output_file.write("\nHamming Loss: " + str(hammingl))
    output_file2.write(str(hammingl) + ", ")

    z1l = sk.metrics.zero_one_loss(y_test, y_pred)
    #print("Zero-one Loss: ", z1l)
    output_file.write("\nZero-one Loss: " + str(z1l))
    output_file2.write(str(z1l)+ "\n")

    return output_path, model,fpr, tpr, thrsh

# This currently does nothing and I have only been using it for experiment with in-built visualizations
def group_plotter(X_test, y_test, ndim, model, output_path):
    #print(output_path)
    output_path = "./results/supervised_methods_evaluation_results/knn/optimized_results/"
    
    #for i in len(y_score_hold):
    #    plt.plot()
    
    counter = 0
    figure1 = plt.figure()
    
    for fig in os.listdir(output_path):
        if fig.endswith(".pickle"):
            fig_object = pickle.load(open(output_path+fig, 'rb'))
            print(fig_object)
            #plt.figure(fig_object)
            #graph_data.append(pickle.load(open(output_path+fig, 'rb')))
        #plt.figure()
    #m_fig, axs = plt.subplots(counter,1)
    #max_l = counter
    #gs = figure1.add_gridspec(1, 3)
    #counter = 0
    #for i in graph_data:
    #    figure1.add_subfigure(gs[0,counter])
    #    counter = counter + 1
    figure1.Figure.show()
    
    return 0
