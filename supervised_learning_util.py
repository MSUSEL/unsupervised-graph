import numpy as np
import pandas as pd

import scipy.spatial.distance
from scipy import spatial

import sklearn as sk
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import *
from sklearn import tree
from sklearn import ensemble
from sklearn import neural_network
from sklearn import naive_bayes
from sklearn import discriminant_analysis
from sklearn.model_selection import cross_val_score

import time
from datetime import timedelta

from concurrent.futures import ThreadPoolExecutor

from supervised_learning_eval import *

import warnings
warnings.filterwarnings("ignore")

X_tr = ""
y_tr = ""
X_te = ""
y_te = "" 


def supervised_learning(output_path, superv_alg_name, ndims_list, emb, clf_type='plain'):    
    fin_fut_objs = []
    exe = ThreadPoolExecutor(8)
    for ndim in ndims_list:
        fin_fut_objs.append(exe.submit(supervised_learning_par_caller, ndim, output_path, superv_alg_name, emb, clf_type))
    
    for obj in range(len(fin_fut_objs)):
        fin_fut_objs[obj] = fin_fut_objs[obj].result()
    print("Learning step completed")
    
    return 0

def supervised_learning_par_caller(ndim, output_path, superv_alg_name, emb, clf_type='plain'):
    print('Dimensions = ', ndim)

    model_path = output_path + emb + '_models/'
    
    train_vector_path = model_path + 'train/train_file_vectors_' + str(ndim) + '.csv'
    train_label_path = model_path + 'train/train_file_labels_' + str(ndim) + '.csv'

    test_vector_path = model_path + 'test/test_file_vectors_' + str(ndim) + '.csv'
    test_label_path = model_path + 'test/test_file_labels_' + str(ndim) + '.csv'
    
    train_df = pd.read_csv(train_label_path)
    y_train = train_df['Label'].tolist() 
    
    train_df = pd.read_csv(test_label_path)
    y_test = train_df['Label'].tolist() 
       
    # Convert 'Benign/Malware' labels to binary (Benign->0 and Malware->1)
    y_train_bin = [1 if idx == 'Malware' else 0 for idx in y_train]
    y_test_bin = [1 if idx == 'Malware' else 0 for idx in y_test]
    
    min_max_scaler = sk.preprocessing.MinMaxScaler()
    #test_scaled = min_max_scaler.fit_transform(ext)
    
    train_vector = pd.read_csv(train_vector_path, header=None).values
    #X_train = StandardScaler().fit_transform(train_vector, y_train_bin)
    X_train = train_vector
    #X_train = min_max_scaler.fit_transform(train_vector)
    #print(np.min(X_train), np.max(X_train))
    #print(np.min(y_train_bin), np.max(y_train_bin))
    
    test_vector = pd.read_csv(test_vector_path, header=None).values
    #X_test = StandardScaler().fit_transform(test_vector, y_test_bin)
    X_test = test_vector
    #X_test = min_max_scaler.fit_transform(test_vector)
        
    #print(len(X_train), len(X_test))
    #print(len(y_train_bin), len(y_test_bin))
        
    X_train_size = len(X_train)
    y_train_size = len(y_train)
    if clf_type == 'optimized':
        for alg in superv_alg_name:
            print("Current Alg: ", alg, " Current Dim: ", ndim)
            model, optimizing_record, training_time = supervised_learning_caller_optimized(alg, X_train, y_train_bin, 
                                                                                               X_test, y_test_bin, emb, True)

            output_path1, model = supervised_methods_evaluation(alg, model, X_train, y_train_bin, X_test, y_test_bin,
                                              ndim, X_train_size, y_train_size,
                                              output_path, emb, training_time,
                                              optimized=True, optimizing_tuple=optimizing_record)
    elif clf_type == 'plain':
        for alg in superv_alg_name:
            print("Current Alg: ", alg, " Current Dim: ", ndim)
            model, training_time = plain_clf_runner(alg, X_train, y_train_bin)
            output_path1, model = supervised_methods_evaluation(alg, model, X_train, y_train_bin, X_test, y_test_bin,
                                                                    ndim, X_train_size, y_train_size,
                                                                    output_path, emb, training_time)
    return 0


def generate_k_nearest_neighbours(X, y, X_train, y_train, emb, pre_optimized = False):
       
    best_rec = ()
    pred_best = 0
    curr_score = 0
    x_dim, y_dim  = X.shape 
    
    if pre_optimized:
        model, best_rec = knn_pre_opt_helper(y_dim, emb)
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

def knn_pre_opt_helper(y_dim, emb):
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


def generate_svc(X, y, X_test, y_test, emb, pre_optimized=False):
    x_dim, y_dim  = X.shape 
    
    if pre_optimized:
        model, best_rec = svc_pre_opt_helper(y_dim, emb)
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

def svc_pre_opt_helper(y_dim, emb):
    if y_dim == 2:
        model = sk.svm.SVC(C=10,
                           kernel='rbf',
                           gamma='scale',
                           tol=.01,
                           random_state = 42,
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


def generate_gaussian_process(X, y, X_test, y_test, emb, pre_optimized = False):
    x_dim, y_dim = X.shape
    if pre_optimized:
        model, best_rec = gb_pre_opt_helper(y_dim, emb)
    
        
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

def gb_pre_opt_helper(y_dim, emb):
    if y_dim == 2:
        kernel_passed = sk.gaussian_process.kernels.RationalQuadratic(alpha=2.5, length_scale=.5)
        model1 = sk.gaussian_process.GaussianProcessClassifier(kernel = kernel_passed,
                                                               optimizer='fmin_l_bfgs_b',
                                                               n_restarts_optimizer=0,
                                                               warm_start=True,
                                                               max_iter_predict=100,
                                                               random_state=42,
                                                               n_jobs=-1) 
        best_rec = ('RationalQuadratic', 'alpha=2.5', 'length_scale=.5')
    elif y_dim == 4:
        kernel_passed = sk.gaussian_process.kernels.RationalQuadratic(alpha=2.5, length_scale=1)
        model1 = sk.gaussian_process.GaussianProcessClassifier(kernel = kernel_passed,
                                                               optimizer='fmin_l_bfgs_b',
                                                               n_restarts_optimizer=0,
                                                               warm_start=True,
                                                               max_iter_predict=100,
                                                               random_state=42,
                                                               n_jobs=-1) 
        best_rec = ('RationalQuadratic', 'alpha=2.5', 'length_scale=1')
    elif y_dim == 8:
        kernel_passed = sk.gaussian_process.kernels.PairwiseKernel(gamma=.8, metric='poly')
        model1 = sk.gaussian_process.GaussianProcessClassifier(kernel = kernel_passed,
                                                               optimizer='fmin_l_bfgs_b',
                                                               n_restarts_optimizer=0,
                                                               warm_start=True,
                                                               max_iter_predict=100,
                                                               random_state=42,
                                                               n_jobs=-1) 
        best_rec = ('PairwiseKernel', 'metric=poly', 'gamma=.8')
    elif y_dim == 16:
        kernel_passed = sk.gaussian_process.kernels.PairwiseKernel(metric='poly', gamma=.653)
        model1 = sk.gaussian_process.GaussianProcessClassifier(kernel = kernel_passed,
                                                               optimizer='fmin_l_bfgs_b',
                                                               n_restarts_optimizer=0,
                                                               warm_start=True,
                                                               max_iter_predict=100,
                                                               random_state=42,
                                                               n_jobs=-1) 
        best_rec = ('PairwiseKernel', 'metric=poly', 'gamma=.8')
    elif y_dim == 32:
        kernel_passed = sk.gaussian_process.kernels.PairwiseKernel(metric='polynomial', gamma=.6538)
        model1 = sk.gaussian_process.GaussianProcessClassifier(kernel = kernel_passed,
                                                               optimizer='fmin_l_bfgs_b',
                                                               n_restarts_optimizer=0,
                                                               warm_start=True,
                                                               max_iter_predict=100,
                                                               random_state=42,
                                                               n_jobs=-1) 
        best_rec = ('PairwiseKernel', 'metric=polynomial', 'gamma=.6538')
    elif y_dim == 64:
        kernel_passed = sk.gaussian_process.kernels.PairwiseKernel(metric='polynomial', gamma=.5)
        model1 = sk.gaussian_process.GaussianProcessClassifier(kernel = kernel_passed,
                                                               optimizer='fmin_l_bfgs_b',
                                                               n_restarts_optimizer=0,
                                                               warm_start=True,
                                                               max_iter_predict=100,
                                                               random_state=42,
                                                               n_jobs=-1) 
        best_rec = ('PairwiseKernel', 'metric=polynomial', 'gamma=.5')
    elif y_dim == 128:
        kernel_passed = sk.gaussian_process.kernels.PairwiseKernel(metric='polynomial', gamma=.5)
        model1 = sk.gaussian_process.GaussianProcessClassifier(kernel = kernel_passed,
                                                               optimizer='fmin_l_bfgs_b',
                                                               n_restarts_optimizer=0,
                                                               warm_start=True,
                                                               max_iter_predict=100,
                                                               random_state=42,
                                                               n_jobs=-1) 
        best_rec = ('PairwiseKernel', 'metric=polynomial', 'gamma=.5')
    elif y_dim == 256:
        kernel_passed = sk.gaussian_process.kernels.PairwiseKernel(metric='polynomial', gamma=1.2)
        model1 = sk.gaussian_process.GaussianProcessClassifier(kernel = kernel_passed,
                                                               optimizer='fmin_l_bfgs_b',
                                                               n_restarts_optimizer=0,
                                                               warm_start=True,
                                                               max_iter_predict=100,
                                                               random_state=42,
                                                               n_jobs=-1) 
        best_rec = ('PairwiseKernel', 'metric=polynomial', 'gamma=1.2')
        
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

def generate_random_forest(X, y, X_train, y_train, emb, pre_optimized = False):
    x_dim, y_dim = X.shape
    if pre_optimized:
        model1, best_rec = rf_pre_opt_helper(y_dim, emb)
        return model1, best_rec
    
    criterion = ['gini', 'entropy', 'log_loss']
    fin_fut_objs = []
    exe = ThreadPoolExecutor(8)
    for n in criterion:
        fin_fut_objs.append(exe.submit(rf_par_helper, n))
    
    for obj in range(len(fin_fut_objs)):
        fin_fut_objs[obj] = fin_fut_objs[obj].result()
    fin_fut_objs = np.array(fin_fut_objs)
    #clf_acc_high = max(fin_fut_objs[:,1])
    clf_acc_high_idx = np.argmax(fin_fut_objs[:,1])
    model1 = fin_fut_objs[clf_acc_high_idx][0]
    best_rec = fin_fut_objs[clf_acc_high_idx][2]
    
    return model1, best_rec

def rf_pre_opt_helper(y_dim, emb):
    
    if emb == 'wlksvd':
        if y_dim == 512:
            model = sk.ensemble.RandomForestClassifier(n_estimators=50,
                                                       criterion='gini',
                                                       min_samples_split=2,
                                                       min_samples_leaf=1,
                                                       max_features='sqrt',
                                                       n_jobs=-1)
            best_rec = ('gini', 50, 2, 1, 'sqrt')
        elif y_dim == 1024:
            model = sk.ensemble.RandomForestClassifier(n_estimators=50,
                                                       criterion='log_loss',
                                                       min_samples_split=2,
                                                       min_samples_leaf=1,
                                                       max_features='sqrt',
                                                       n_jobs=-1)
            best_rec = ('log_loss', 50, 2, 1, 'sqrt')
        elif y_dim == 2048:
            model = sk.ensemble.RandomForestClassifier(n_estimators=50,
                                                       criterion='log_loss',
                                                       min_samples_split=2,
                                                       min_samples_leaf=1,
                                                       max_features='sqrt',
                                                       n_jobs=-1)
            best_rec = ('log_loss', 50, 2, 1, 'sqrt')
        elif y_dim == 4096:
            model = sk.ensemble.RandomForestClassifier(n_estimators=50,
                                                       criterion='log_loss',
                                                       min_samples_split=2,
                                                       min_samples_leaf=1,
                                                       max_features='sqrt',
                                                       n_jobs=-1)
            best_rec = ('log_loss', 50, 2, 1, 'sqrt')      
        
    elif emb == 'd2v':
        if y_dim == 2:
            model = sk.ensemble.RandomForestClassifier(n_estimators=150,
                                                       criterion='gini',
                                                       min_samples_split=2,
                                                       min_samples_leaf=1,
                                                       max_features='sqrt',
                                                       n_jobs=-1)
            best_rec = ('gini', 150, 2, 1, 'sqrt')
        elif y_dim == 4:
            model = sk.ensemble.RandomForestClassifier(n_estimators=150,
                                                       criterion='entropy',
                                                       min_samples_split=5,
                                                       min_samples_leaf=2,
                                                       max_features='sqrt',
                                                       n_jobs=-1)
            best_rec = ('entropy', 150, 5, 2, 'sqrt')
        elif y_dim == 8:
            model = sk.ensemble.RandomForestClassifier(n_estimators=150,
                                                       criterion='log_loss',
                                                       min_samples_split=6,
                                                       min_samples_leaf=1,
                                                       max_features='sqrt',
                                                       n_jobs=-1)
            best_rec = ('log_loss', 150, 6, 1, 'sqrt')
        elif y_dim == 16:
            model = sk.ensemble.RandomForestClassifier(n_estimators=150,
                                                       criterion='gini',
                                                       min_samples_split=3,
                                                       min_samples_leaf=2,
                                                       max_features='log2',
                                                       n_jobs=-1)
            best_rec = ('gini', 150, 3, 2, 'log2')
        elif y_dim == 32:
            model = sk.ensemble.RandomForestClassifier(n_estimators=150,
                                                       criterion='entropy',
                                                       min_samples_split=5,
                                                       min_samples_leaf=1,
                                                       max_features='sqrt',
                                                       n_jobs=-1)
            best_rec = ('entropy', 150, 5, 1, 'sqrt')
        elif y_dim == 64:
            model = sk.ensemble.RandomForestClassifier(n_estimators=150,
                                                       criterion='entropy',
                                                       min_samples_split=2,
                                                       min_samples_leaf=2,
                                                       max_features='sqrt',
                                                       n_jobs=-1)
            best_rec = ('entropy', 150, 2, 2, 'sqrt')
        elif y_dim == 128:
            model = sk.ensemble.RandomForestClassifier(n_estimators=150,
                                                       criterion='entropy',
                                                       min_samples_split=3,
                                                       min_samples_leaf=3,
                                                       max_features='log2',
                                                       n_jobs=-1)
            best_rec = ('entropy', 150, 3, 3, 'log2')
        elif y_dim == 256:
            model = sk.ensemble.RandomForestClassifier(n_estimators=150,
                                                       criterion='gini',
                                                       min_samples_split=3,
                                                       min_samples_leaf=1,
                                                       max_features='sqrt',
                                                       n_jobs=-1)
            best_rec = ('gini', 150, 3, 1, 'sqrt')
        
    return model, best_rec
    
def rf_par_helper(criterion):
    n_est = [50,100,150]
    max_feature = ['sqrt', 'log2' ] #, None]
    min_samples_split = [2,3,5,6 ]# ,10]
    min_samples_leaf = [1,2,3,4,5 ]  #,10]
    pred_best = 0
    for i in n_est:
        for j in max_feature:
            for k in min_samples_split:
                for l in min_samples_leaf:
                    model = sk.ensemble.RandomForestClassifier(n_estimators=i,
                                                               criterion=criterion,
                                                               min_samples_split=k,
                                                               min_samples_leaf=l,
                                                               max_features=j,
                                                               n_jobs=-1)
                    model_scores = cross_val_score(model, X_tr, y_tr, cv = 5, n_jobs = -1 )
                    curr_score = model_scores.mean()
                    if curr_score > pred_best:
                        best_rec = (curr_score, criterion, i, k, l, j)
                        model1 = model
                        pred_best = curr_score 
                    print(criterion, curr_score, i, k, l, j)
    return model1, pred_best, best_rec

def generate_hist_gradient_boosting_clf(X, y, X_test, y_test, emb, pre_optimized = False):
    x_dim, y_dim = X.shape
    if pre_optimized:
        model1, best_rec = hgbc_pre_opt_helper(y_dim, emb)
        return model1, best_rec
    
    max_iter = [ 100, 125, 135]
    fin_fut_objs = []
    exe = ThreadPoolExecutor(8)
    for n in max_iter:
        fin_fut_objs.append(exe.submit(hgbc_par_helper, n))
    
    for obj in range(len(fin_fut_objs)):
        fin_fut_objs[obj] = fin_fut_objs[obj].result()
    fin_fut_objs = np.array(fin_fut_objs)
    #clf_acc_high = max(fin_fut_objs[:,1])
    clf_acc_high_idx = np.argmax(fin_fut_objs[:,1])
    model1 = fin_fut_objs[clf_acc_high_idx][0]
    best_rec = fin_fut_objs[clf_acc_high_idx][2]
    return model1, best_rec

def hgbc_pre_opt_helper(y_dim, emb):
    
    if emb == 'wlksvd':
        if y_dim == 512:
            model = sk.ensemble.HistGradientBoostingClassifier(max_iter = 125,
                                                               learning_rate = .1,
                                                               l2_regularization = 0,
                                                               tol = 1*10**-9,
                                                               random_state = 42)
            best_rec = (125, .1, 0, 1*10**-9, 42)
        elif y_dim == 1024:
            model = sk.ensemble.HistGradientBoostingClassifier(max_iter = 125,
                                                               learning_rate = .1,
                                                               l2_regularization = 0,
                                                               tol = 1*10**-9,
                                                               random_state = 42)
            best_rec = (125, .1, 0, 1*10**-9, 42)
        elif y_dim == 2048:
            model = sk.ensemble.HistGradientBoostingClassifier(max_iter = 135,
                                                               learning_rate = .1,
                                                               l2_regularization = 0,
                                                               tol = 1*10**-9,
                                                               random_state = 42)
            best_rec = (135, .1, 0, 1*10**-9, 42)
        elif y_dim == 4096:
            model = sk.ensemble.HistGradientBoostingClassifier(max_iter = 125,
                                                               learning_rate = .3,
                                                               l2_regularization = 2,
                                                               tol = 1*10**-9,
                                                               random_state = 42)
            best_rec = (125, .3, 2, 1*10**-9, 42)
        
    elif emb == 'd2v':
        if y_dim == 2:
            model = sk.ensemble.HistGradientBoostingClassifier(max_iter = 125,
                                                               learning_rate = .3,
                                                               l2_regularization = 0,
                                                               tol = 1*10**-9,
                                                               random_state = 42)
            best_rec = (125, .3, 0, 1*10**-9, 42)
        elif y_dim == 4:
            model = sk.ensemble.HistGradientBoostingClassifier(max_iter = 125,
                                                               learning_rate = .3,
                                                               l2_regularization = 0,
                                                               tol = 1*10**-9,
                                                               random_state = 42)
            best_rec = (125, .3, 0, 1*10**-9, 42)
        elif y_dim == 8:
            model = sk.ensemble.HistGradientBoostingClassifier(max_iter = 125,
                                                               learning_rate = .3,
                                                               l2_regularization = 0,
                                                               tol = 1*10**-9,
                                                               random_state = 42)
            best_rec = (125, .3, 0, 1*10**-9, 42)
        elif y_dim == 16:
            model = sk.ensemble.HistGradientBoostingClassifier(max_iter = 125,
                                                               learning_rate = .3,
                                                               l2_regularization = 0,
                                                               tol = 1*10**-9,
                                                               random_state = 42)
            best_rec = (125, .3, 0, 1*10**-9, 42)
        elif y_dim == 32:
            model = sk.ensemble.HistGradientBoostingClassifier(max_iter = 125,
                                                               learning_rate = .3,
                                                               l2_regularization = 0,
                                                               tol = 1*10**-9,
                                                               random_state = 42)
            best_rec = (125, .3, 0, 1*10**-9, 42)
        elif y_dim == 64:
            model = sk.ensemble.HistGradientBoostingClassifier(max_iter = 125,
                                                               learning_rate = .3,
                                                               l2_regularization = 0,
                                                               tol = 1*10**-9,
                                                               random_state = 42)
            best_rec = (125, .3, 0, 1*10**-9, 42)
        elif y_dim == 128:
            model = sk.ensemble.HistGradientBoostingClassifier(max_iter = 125,
                                                               learning_rate = .3,
                                                               l2_regularization = 0,
                                                               tol = 1*10**-9,
                                                               random_state = 42)
            best_rec = (125, .3, 0, 1*10**-9, 42)
        elif y_dim == 256:
            model = sk.ensemble.HistGradientBoostingClassifier(max_iter = 125,
                                                               learning_rate = .3,
                                                               l2_regularization = 0,
                                                               tol = 1*10**-9,
                                                               random_state = 42)
            best_rec = (125, .3, 0, 1*10**-9, 42)
    return model, best_rec

def hgbc_par_helper(max_iter):
    pred_best = 0
    learning_rate = [.1, .2, .3]
    l2_regularization = [ 0, 1, 2]
    tol = [ 1*10**-9, 1*10**-7]
    
    for i in learning_rate:
        for j in l2_regularization:
            for k in tol:
                model = sk.ensemble.HistGradientBoostingClassifier(max_iter = max_iter,
                                                                   learning_rate = i,
                                                                   l2_regularization = j,
                                                                   tol = k,
                                                                   random_state = 42)
                model_scores = cross_val_score(model, X_tr, y_tr, cv = 5, n_jobs = -1 )
                curr_score = model_scores.mean()
                if curr_score > pred_best:
                    best_rec = (curr_score, max_iter, i, j, k)
                    model1 = model
                    pred_best = curr_score 
                print(curr_score, max_iter, i, j, k)
    
    return model1, pred_best, best_rec

def generate_linear_svc(X, y, X_train, y_train, emb, pre_optimized = False):
    x_dim, y_dim = X.shape
    
    if pre_optimized:
        model1, best_rec = lsvc_pre_opt_helper(y_dim, emb)
        return model1, best_rec
    
    loss = ['hinge', 'squared_hinge']
    fin_fut_objs = []
    exe = ThreadPoolExecutor(8)
    counter = 1
    for n in loss:
        print("Num of par jobs in inner parallelization: ", counter)
        counter = counter + 1
        fin_fut_objs.append(exe.submit(lsvc_par_helper, n))
    
    for obj in range(len(fin_fut_objs)):
        fin_fut_objs[obj] = fin_fut_objs[obj].result()
    fin_fut_objs = np.array(fin_fut_objs)
    clf_acc_high_idx = np.argmax(fin_fut_objs[:,1])
    model1 = fin_fut_objs[clf_acc_high_idx][0]
    best_rec = fin_fut_objs[clf_acc_high_idx][2]
 
    return model1, best_rec 

def lsvc_pre_opt_helper(y_dim, emb):
    if y_dim == 512:
        model = sk.svm.LinearSVC(penalty='l2',
                                 loss='squared_hinge',
                                 dual=True,
                                 tol=.01,
                                 C=1.5,
                                 random_state=42)
        best_rec = ('squared_hinge', 0.01, 1.5)
    elif y_dim == 1024:
        model = sk.svm.LinearSVC(penalty='l2',
                                 loss='squared_hinge',
                                 dual=True,
                                 tol=.01,
                                 C=1.3,
                                 random_state=42)
        best_rec = ( 'squared_hinge', 0.01, 1.3)
    elif y_dim == 2048:
        model = sk.svm.LinearSVC(penalty='l2',
                                 loss='squared_hinge',
                                 dual=True,
                                 tol=.01,
                                 C=.5,
                                 random_state=42)
        best_rec = ('squared_hinge', 0.01, 0.5)
    elif y_dim == 4096:
        model = sk.svm.LinearSVC(penalty='l2',
                                 loss='squared_hinge',
                                 dual=True,
                                 tol=.01,
                                 C=1.5,
                                 random_state=42)
        best_rec = ('squared_hinge', 0.01, 1.5)

    return model, best_rec

def lsvc_par_helper(loss):
    pred_best = 0
    #penalty = ['l1', 'l2']
    penalty = ['l2']
    tol = [ 1*10**-2, 1*10**-3, 1*10**-4, 1*10**-5 ]
    C = [.5, .7, 1, 1.3, 1.5]
    if loss == 'hinge':
        for i in tol:
            for j in C:
                model = sk.svm.LinearSVC(penalty='l2',
                                         loss=loss,
                                         dual=True,
                                         tol=i,
                                         C=j,
                                         random_state=42)
            model_scores = cross_val_score(model, X_tr, y_tr, cv = 5, n_jobs = -1 )
            curr_score = model_scores.mean()
            if curr_score > pred_best:
                best_rec = (curr_score, loss, i, j)
                model1 = model
                pred_best = curr_score 
            print(curr_score, loss, 'l2', i, j)
                
    elif loss == 'squared_hinge':
        for k in penalty:
            for i in tol:
                for j in C:
                    model = sk.svm.LinearSVC(penalty=k,
                                             loss=loss,
                                             dual=True,
                                             tol=i,
                                             C=j,
                                             random_state=42)
                    model_scores = cross_val_score(model, X_tr, y_tr, cv = 5, n_jobs = -1 )
                    curr_score = model_scores.mean()
                    if curr_score > pred_best:
                        best_rec = (curr_score, loss, i, j)
                        model1 = model
                        pred_best = curr_score 
                    print(curr_score, loss, k, i, j)
                
    return model1, pred_best, best_rec

def supervised_learning_caller_optimized(alg, X_train, y_train, X_test, y_test, emb, pre_optimized):
    global X_tr, y_tr, X_te, y_te
    X_tr = X_train
    y_tr = y_train
    X_te = X_test
    y_te = y_test
    
    if alg == 'knn':
        start_time = time.monotonic()
        model, best_rec = generate_k_nearest_neighbours(X_train, y_train, X_test, y_test, emb, pre_optimized)
        training_time = timedelta(seconds=time.monotonic()-start_time)
    elif alg == 'svc':
        start_time = time.monotonic()
        model, best_rec = generate_svc(X_train, y_train, X_test, y_test, emb, pre_optimized)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'gp':
        start_time = time.monotonic()
        model, best_rec = generate_gaussian_process(X_train, y_train, X_test, y_test, emb, pre_optimized)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'rf':
        start_time = time.monotonic()
        model, best_rec = generate_random_forest(X_train, y_train, X_test, y_test, emb, pre_optimized)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'lsvc':
        start_time = time.monotonic()
        model, best_rec = generate_linear_svc(X_train, y_train, X_test, y_test, emb, pre_optimized)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'hgbc':
        start_time = time.monotonic()
        model, best_rec = generate_hist_gradient_boosting_clf(X_train, y_train, X_test, y_test, emb, pre_optimized)
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
        model = sk.svm.NuSVC(probability=True)
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
    elif alg == 'ridge':
        start_time = time.monotonic()
        model = sk.linear_model.RidgeClassifier()
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg ==  'pa':
        start_time = time.monotonic()
        model = sk.linear_model.PassiveAggressiveClassifier()
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg ==  'sgd':
        start_time = time.monotonic()
        model = sk.linear_model.SGDClassifier()
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg ==  'perc':
        start_time = time.monotonic()
        model = sk.linear_model.Perceptron()
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'gbc':
        start_time = time.monotonic()
        model = sk.ensemble.GradientBoostingClassifier()
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'etc':
        start_time = time.monotonic()
        model = sk.ensemble.ExtraTreesClassifier()
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'hgbc':
        start_time = time.monotonic()
        model = sk.ensemble.HistGradientBoostingClassifier()
        training_time = timedelta(seconds=time.monotonic() - start_time)
    
    return model, training_time
