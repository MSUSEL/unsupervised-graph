import time

import networkx
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler as min_max_scaler

from g2v_util import *
from cluster_util import *
from supervised_learning_util import *
from supervised_learning_eval import *
from create_embedding_comp import *

from concurrent.futures import ThreadPoolExecutor
import gc
import os

def create_CFG_datastet(prog_path, cfg_path, n_malware, n_benign):
    ''' Create the CFG dataset from set of binary files

    '''
    malware_binary_path = prog_path + 'malware/'
    benign_binary_path = prog_path + 'benign/'

    malware_cfg_path = cfg_path + 'Malware_CFG/'
    benign_cfg_path = cfg_path + 'Benign_CFG/'

    createSaveCFG(malware_binary_path, malware_cfg_path, n_malware)
    createSaveCFG(benign_binary_path, benign_cfg_path, n_benign)

if __name__ == "__main__":
    print('Binary file analysis')

    # define paths for the binary files, CFG files, and Output files
    prog_path = './Binary_data/'
    cfg_path = './data/CFG_dataset/'
    
    #here for use on my local machine can remove
    #cfg_path = 'D:/supervisedLearningmeths/CFG_dataset/'
    
    output_path = './results/'
    info_path = '/data/CFG_dataset/class_labels_no_dupes.csv'

    ######## 1. Creating CFGs from binary files ############
    print('******************* STEP: 1 *******************')
    # Skip this step if you already have the CFG dataset
    # Note: The binary files are not provided

    ## Training binary files
    train_prog_path = prog_path + 'Train/'
    train_cfg_path = cfg_path + 'Train_CFG/'
    n_malware = 3000 # Maximum 3000,  for quick run use 300
    n_benign = 3000  # Maximum 3000,  for quick run use 300
    #create_CFG_datastet(train_prog_path, cfg_path, n_malware, n_benign)
    
    ## Testing binary files
    test_prog_path = prog_path + 'Test/'
    test_cfg_path = cfg_path + 'Test_CFG/'
    n_test_malware = 1000 # Maximum 1000, for quick run use 100
    n_test_benign = 1000 # Maximum 1000, for quick run use 100
    #create_CFG_datastet(test_prog_path, cfg_path, n_malware, n_benign)
    
    ######## 2. Create vector representation for CFGs using 'Graph2Vec' graph embedding. ############
    print('******************* STEP: 2 *******************')

    # Graph2Vec dimensions
    g2v_ndims_list = [2, 4, 8, 16, 32, 64, 128, 256]
    
    wlksvd_ndims_list = [ 512, 1024, 2048, 4096 ]
    
    # The hyper parameters for embedding is inside the function 
    
    embed_list = [ 'wlksvd', 'd2v']
    
    #model_path = create_embedding(embed_list, cfg_path, output_path, ndims_list, n_malware, n_benign, n_test_malware, n_test_benign, isTrain = True)
    
    ####### 3. Unsupervised clustering algorithm training with hold-out validation. ##################
    print('******************* STEP: 3 *******************')
    # Unsupervised clustering algorithms
    #cluster_alg_name = ['Kmeans', 'spectral', 'Aggloromative', 'DBSCAN']
    #cluster_alg_name = ['Kmeans', 'Aggloromative']
    # The hyper parameters for each clustering method is inside the function

    #clustering_training(output_path, embed_list, cluster_alg_name, g2v_ndims_list, wlksvd_ndims_list)

    ####### 4. Cluster prediction for Test dataset ##################
    print('******************* STEP: 4 *******************')
    #cluster_prediction(test_cfg_path, output_path, param, cluster_alg_name, ndims_list)

    ####### 5. Supervised learning procedures for Test dataset ##################
    print('******************* STEP: 5 *******************')
    
    #all implemented classifiers( mnb, compnb, catnb, and potentially gp are not functional if data contains negative values, mlp is also not functional)
    #superv_alg_name = [ 'knn', 'rnn', 'lsvc', 'nsvc', 'svc', 'gp', 'dt', 'rf', 'ada', 'gnb', 'mnb', 
    #                    'compnb', 'catnb' ,'qda', 'lda', 'mlp', 'ridge', 'pa', 'sgd', 'perc', 'etc', 'hgbc', 'gbc']
    
    d2v_superv_alg_name = [ 'svc' ] # , 'rf', 'hgbc' ]
    
    wlksvd_superv_alg_name = [ 'lsvc' ] # , 'rf', 'hgbc' ]
    
    clf_type1 = 'plain'
    clf_type2 = 'optimized'
    
    for emb in embed_list:
        if emb == 'd2v':
            ndims_list = g2v_ndims_list 
            superv_alg_name = d2v_superv_alg_name
        elif emb == 'wlksvd':
            ndims_list = wlksvd_ndims_list
            superv_alg_name = wlksvd_superv_alg_name
            
        print("Running Classifiers on " + classifier_name_expand(emb) + " embedded vectors" )
    
        print("Plain classifier check")
        supervised_learning(output_path, superv_alg_name, ndims_list, emb, clf_type1)
    
        print("Optimized classifier check")
        supervised_learning(output_path, superv_alg_name, ndims_list, emb, clf_type2)
        
    print("Running Plotting Functions")
    for emb in embed_list:
        if emb == 'd2v':
            ndims_list = g2v_ndims_list 
            superv_alg_name = d2v_superv_alg_name
        elif emb == 'wlksvd':
            ndims_list = wlksvd_ndims_list
            superv_alg_name = wlksvd_superv_alg_name
            
        group_plotter_by_dim(superv_alg_name, emb, ndims_list, output_path, 'optimized')
        group_plotter_by_dim(superv_alg_name, emb, ndims_list, output_path, 'plain')
        
        group_plotter_by_alg(superv_alg_name, emb, ndims_list, output_path, 'optimized')
        group_plotter_by_alg(superv_alg_name, emb, ndims_list, output_path, 'plain')
    
    print('******************* Process Finished *******************')
