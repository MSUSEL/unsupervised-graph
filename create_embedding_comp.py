from reader import *
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from karateclub.graph_embedding import Graph2Vec, GL2Vec, SF, IGE
import timeit
import time
from datetime import timedelta

from WL_KSVD import *
from g2v_util import *
from supervised_learning_eval import embedding_timing_writer

import pandas as pd
import numpy as np
import os
import gc
from joblib import dump, load
import pickle

import psutil

def create_embedding(embedding_list, cfg_path, output_path, ndims_list, n_malware_train, n_benign_train, n_malware_test, n_benign_test, isTrain):
    #FINISH CLEANING UP EMBEDDING ROUTINES
    for embedding in embedding_list:
        if embedding == 'g2v':
            timing_path = output_path + 'g2v_embedding_timing/'
            model_output_path, fit_time, inf_time = graph2vec_emb(cfg_path, output_path, ndims_list, n_malware_train, n_benign_train, isTrain)
        elif embedding == 'wlksvd':
            timing_path = output_path + 'wlksvd_embedding_timing/'
            model_output_path, fit_time, inf_time = wlksvd_emb(cfg_path, output_path, ndims_list, n_malware_train, n_benign_train, n_malware_test, n_benign_test, isTrain)
            
        embedding_timing_writer(timing_path, fit_time, inf_time, isTrain)
    return model_output_path
    
def graph2vec_emb(cfg_path, output_path, ndims_list, n_malware_train, n_benign_train, n_malware_test, n_benign_test, isTrain=True):
    if isTrain:
        malware_cfg_path = cfg_path + 'Train_CFG/Malware_CFG/'
        benign_cfg_path = cfg_path + 'Train_CFG/Benign_CFG/'
        n_percent_train = .2 # percentage for vocabulary  training (20% validation)
    else:
        malware_cfg_path = cfg_path + 'Test_CFG/Malware_CFG/'
        benign_cfg_path = cfg_path + 'Test_CFG/Benign_CFG/'
        n_percent_train = 0.0
    
    # Load .gpickle CFG files for training set
    Malware_graphs, Malware_names = loadCFG(malware_cfg_path, n_malware)
    Benign_graphs, Benign_names = loadCFG(benign_cfg_path, n_benign)

    ## Parameters
    param = WL_parameters()
    param._set_seed()
    
    if isTrain:
    ## Train divide for vocabulary training graphs
        vocab_train_graphs, inp_graphs, vocab_train_labels, inp_labels, n_vocab_train, n_inp, vocab_train_names, inp_names = \
            train_test_divide(Malware_graphs, Benign_graphs, Malware_names, Benign_names, n_malware_train, n_benign_train,
                              n_percent_train)

        ########## Training Graph2Vec* Model ##############
        
        # Train and save Graph2Vec model
        start_time = time.monotonic()
        train_G2V_model(vocab_train_graphs, vocab_train_labels, vocab_train_names, param, ndims_list, save_model=True, output_path=output_path)
        fit_time = timedelta(seconds=time.monotonic() - start_time)
        
        if 'vocab_train_graphs' in os.environ:
            del (vocab_train_graphs)

        if 'vocab_train_labels' in os.environ:
            del os.environ['vocab_train_labels']

        if 'vocab_train_names' in os.environ:
            del os.environ['vocab_train_names']

        if 'd2v_model' in os.environ:
            del os.environ['d2v_model']

        ######### Inferencing the vector for  data ################

        # Graph2Vec* inference
        #print('Graph2Vec inference')
        # test_vector, test_vector_labels, test_vector_names = inferG2V(test_graphs, test_labels, test_names, param)
    
        # Create WL hash word documents for testing set
        print('Creating WL hash words for remaining training set')
        inp_documents = createWLhash(inp_graphs, param)
    
        if 'inp_graphs' in os.environ:
            del inp_graphs
    
    else:
        empty_set, inp_graphs, empty_labels, inp_labels, empty_num, n_inp, empty_names, inp_names = \
            train_test_divide(Malware_graphs, Benign_graphs, Malware_names, Benign_names, n_malware_test, n_benign_test,
                              n_percent_train)
    
        # Save memory by removing unnecessary  variables
        if 'Malware_graphs' in os.environ:
            del(Malware_graphs)

        if 'Benign_graphs' in os.environ:
            del(Benign_graphs)

        if 'Malware_names' in os.environ:
            del(Malware_names)

        if 'Benign_names' in os.environ:
            del(Benign_names)

        gc.collect()
       
        print('Creating WL hash words for test set')
        inp_documents = createWLhash(inp_graphs, param)
    
        if 'inp_graphs' in os.environ:
            del inp_graphs
                
    for ndim in ndims_list:
        print('Dimensions = ', ndim)

        ## Parameters
        param = WL_parameters(dimensions=ndim)
        param._set_seed()

        # model path
        model_path = output_path + 'd2v_models/'
        if isTrain:
            model_out_path = model_path + 'train/'
        else:
            model_out_path = model_path + 'test/'
        
        os.makedirs(os.path.dirname(model_out_path), exist_ok=True)        
         
        model_name = (model_path + 'd2v_model_' + str(param.dimensions) + '.model')
            
        # writing data for train set
        
        #print(model_train_name, model_test_name)
        try:
            d2v_model = Doc2Vec.load(model_name)
        except Exception as e:
            print("ERROR!!!!!!! : %s" % e)
            
        print("Training Set Embedding")
        ## Shuffling of the data
        print('Shuffling data')
        inp_corpus, inp_vector_labels, inp_vector_names = DocShuffle(inp_documents, inp_labels, inp_names)

        ## Doc2Vec inference
        print('Doc2Vec inference')
        start_time = time.monotonic()
        inp_vector = np.array([d2v_model.infer_vector(d.words) for d in inp_corpus])
        infer_time = timedelta(seconds=time.monotonic() - start_time)
        
        if isTrain:
            vector_out_path = Path(model_out_path + 'train_file_vectors_' + str(param.dimensions) + '.csv')
            vector_out_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(inp_vector).to_csv(vector_out_path, header=None, index=None)

            label_out_path = Path(model_out_path + 'train_file_labels_' + str(param.dimensions) + '.csv')
            label_out_path.parent.mkdir(parents=True, exist_ok=True)
            train_df = pd.DataFrame({'name': inp_vector_names, 'Label': inp_vector_labels})
            train_df.to_csv(label_out_path)

            ## Visualizing
            print('Visualizations')
            twoD_tsne_vector, fig = TSNE_2D_plot(inp_vector, inp_vector_labels, n_inp, param.dimensions, 'Graph2Vec',
                                                 return_plot=True)

            fig_name = model_out_path + '/' + 'train_vector_' + str(param.dimensions) + '-dims.png'
            fig.savefig(fig_name)
            plt.clf()
 
        else:
            vector_out_path = Path(model_out_path + 'test_file_vectors_' + str(param.dimensions) + '.csv')
            vector_out_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(inp_vector).to_csv(vector_out_path, header=None, index=None)

            label_out_path = Path(model_out_path + 'test_file_labels_' + str(param.dimensions) + '.csv')
            label_out_path.parent.mkdir(parents=True, exist_ok=True)
            train_df = pd.DataFrame({'name': inp_vector_names, 'Label': inp_vector_labels})
            train_df.to_csv(label_out_path)

            ## Visualizing
            print('Visualizations')
            twoD_tsne_vector, fig = TSNE_2D_plot(inp_vector, inp_vector_labels, n_inp, param.dimensions, 'Graph2Vec',
                                                 return_plot=True)

            fig_name = model_out_path + '/' + 'test_vector_' + str(param.dimensions) + '-dims.png'
            fig.savefig(fig_name)
            plt.clf()

    return model_out_path, fit_time, infer_time

def wlksvd_emb(cfg_path, output_path, ndims_list, n_malware_train, n_benign_train,
                   n_malware_test, n_benign_test, isTrain=True):
    
    model_path = output_path + "wlksvd_models"

    malware_cfg_path = cfg_path + 'Train_CFG/Malware_CFG/'
    benign_cfg_path = cfg_path + 'Train_CFG/Benign_CFG/'
    
    Malware_graphs, Malware_names = loadCFG(malware_cfg_path, n_malware_train)
    Benign_graphs, Benign_names = loadCFG(benign_cfg_path, n_benign_train)
    
    n_percent_train = 0.2
    vocab_train_graphs, inp_graphs, vocab_train_labels, inp_labels, vocab_train_num, n_inp, vocab_train_names, inp_names = \
            train_test_divide(Malware_graphs, Benign_graphs, Malware_names, Benign_names, n_malware_train, n_benign_train,
                              n_percent_train)
    
    vocab_train_graphs, vocab_train_labels, vocab_train_names = DocShuffle( vocab_train_graphs, vocab_train_labels, vocab_train_names)
    
    inp_graphs, inp_labels, inp_names = DocShuffle( inp_graphs, inp_labels, inp_names)
    
    for ndim in ndims_list:
        print("Dimensions(train set): ", ndim)
        model_name = (model_path + '/' + 'wlksvd_model_' + str(ndim) + '.model')
        model = WL_KSVD(dimensions=ndim, n_vocab=1000,
                                        n_non_zero_coefs=int(np.ceil(ndim / 10)))
        print("Fitting model")
        
        start_time = time.monotonic()
        model.fit(vocab_train_graphs)
        fit_time = timedelta(seconds=time.monotonic() - start_time)
        
        model_out_path = Path(model_path + '/wlksvd_model_' + str(ndim) + '.model')
        model_out_path.parent.mkdir(parents=True, exist_ok=True)
        X_vec = model.get_embedding()
        pd.DataFrame(X_vec).to_csv(model_out_path, header=None, index=None)
        print("Inferring graphs")
        
        start_time = time.monotonic()
        X_vec = model.infer(inp_graphs)
        infer_time = timedelta(seconds=time.monotonic() - start_time)
        
        vector_out_path = Path(model_path + '/train/' + 'train_file_vectors_' + str(ndim) + '.csv') 
        vector_out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(X_vec).to_csv(vector_out_path, header=None, index=None)
        
        label_out_path = Path(model_path + '/train/' + 'train_file_labels_' + str(ndim) + '.csv')
        label_out_path.parent.mkdir(parents=True, exist_ok=True)
        train_df = pd.DataFrame({'name': inp_names, 'Label': inp_labels})
        train_df.to_csv(label_out_path)
        
        ## Visualizing
        print('Visualizations')
        twoD_tsne_vector, fig = TSNE_2D_plot(X_vec, inp_labels, n_inp, ndim, 'WLKSVD',
                                                 return_plot=True)

        fig_name = model_path + '/train/train_vector_' + str(ndim) + '-dims.png'
        fig.savefig(fig_name)
        plt.clf()
    
    malware_cfg_path = cfg_path + 'Test_CFG/Malware_CFG/'
    benign_cfg_path = cfg_path + 'Test_CFG/Benign_CFG/'
    
    Malware_graphs, Malware_names = loadCFG(malware_cfg_path, n_malware_test)
    Benign_graphs, Benign_names = loadCFG(benign_cfg_path, n_benign_test)
    
    n_percent_train = 0.0
    empty_graphs, inp2_graphs, empty_labels, inp2_labels, empty_num, n_inp2, empty_names, inp2_names = \
            train_test_divide(Malware_graphs, Benign_graphs, Malware_names, Benign_names, n_malware_test, n_benign_test,
                              n_percent_train)
    
    inp2_graphs, inp2_labels, inp2_names = DocShuffle(inp2_graphs, inp2_labels, inp2_names)     
    for ndim in ndims_list:
        print("Dimensions(test set): ", ndim)
        model = WL_KSVD(dimensions=ndim, n_vocab=1000,
                                        n_non_zero_coefs=int(np.ceil(ndim / 10)))
        print("Fitting model")
        
        print("Inferring graphs")
        start_time = time.monotonic()
        X_vec = model.infer(inp2_graphs)
        infer_time = timedelta(seconds=time.monotonic() - start_time)
        
        vector_out_path = Path(model_path + '/test/' + 'test_file_vectors_' + str(ndim) + '.csv')
        vector_out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(X_vec).to_csv(vector_out_path, header=None, index=None)
        label_out_path = Path(model_path + '/test/' + 'test_file_labels_' + str(ndim) + '.csv')
        label_out_path.parent.mkdir(parents=True, exist_ok=True)
        train_df = pd.DataFrame({'name': inp2_names, 'Label': inp2_labels})
        train_df.to_csv(label_out_path)
        
        ## Visualizing
        print('Visualizations')
        twoD_tsne_vector, fig = TSNE_2D_plot(X_vec, inp2_labels, n_inp2, ndim, 'WLKSVD',
                                                 return_plot=True)

        fig_name = model_path + '/test/test_vector_' + str(ndim) + '-dims.png'
        fig.savefig(fig_name)
        plt.clf()
    return model_path, fit_time, infer_time


#currently memming out on the full benign/malicious dataset
def gl2vec_emb(cfg_path, output_path, ndims_list, n_malware_train, n_benign_train,
                   n_malware_test, n_benign_test, isTrain=True):
    
    #print('1st')
    #print(psutil.virtual_memory()[2])
    #print(psutil.virtual_memory()[3]/1000000000)
    
    model_path = output_path + "gl2v_models"

    malware_cfg_path = cfg_path + 'Train_CFG/Malware_CFG/'
    benign_cfg_path = cfg_path + 'Train_CFG/Benign_CFG/'
    
    Malware_graphs, Malware_names = loadCFG(malware_cfg_path, n_malware_train)
    Benign_graphs, Benign_names = loadCFG(benign_cfg_path, n_benign_train)
    
    n_percent_train = 0.2
    vocab_train_graphs, inp_graphs, vocab_train_labels, inp_labels, vocab_train_num, n_inp, vocab_train_names, inp_names = \
            train_test_divide(Malware_graphs, Benign_graphs, Malware_names, Benign_names, n_malware_train, n_benign_train,
                              n_percent_train)
    
    vocab_train_graphs, vocab_train_labels, vocab_train_names = DocShuffle( vocab_train_graphs, vocab_train_labels, vocab_train_names)
    
    inp_graphs, inp_labels, inp_names = DocShuffle( inp_graphs, inp_labels, inp_names)
           
    del Malware_graphs
        
    del Malware_names
        
    del Benign_names
        
    del Benign_graphs
    
    gc.collect()
    
    for ndim in ndims_list:
        print("Dimensions(train set): ", ndim)
        model_name = (model_path + '/' + 'gl2v_model_' + str(ndim) + '.model')
        model = GL2Vec(dimensions=ndim, workers = 8)
        print("Fitting model")
        model.fit(vocab_train_graphs)
        model_out_path = Path(model_path + '/gl2v_model_' + str(ndim) + '.model')
        model_out_path.parent.mkdir(parents=True, exist_ok=True)
        X_vec = model.get_embedding()
        pd.DataFrame(X_vec).to_csv(model_out_path, header=None, index=None)
        
        #print(os.environ[X_vec], os.environ[vocab_train_graphs],  os.environ[vocab_train_labels],  os.environ[vocab_train_names], os.environ[vocab_train_num])
        
        del X_vec
        
        gc.collect()
        
        print("4th")
        print(psutil.virtual_memory()[2])
        print(psutil.virtual_memory()[3]/1000000000)
        
        print("Inferring graphs")
        X_vec = np.memmap("X_vec.arr", mode='w+', dtype=np.uint8, shape=(n_inp, ndim))
        X_vec[:] = model.infer(inp_graphs)
        #X_vec = model.infer(inp_graphs)
        
        vector_out_path = Path(model_path + '/train/' + 'train_file_vectors_' + str(ndim) + '.csv') 
        vector_out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(X_vec).to_csv(vector_out_path, header=None, index=None)
        
        label_out_path = Path(model_path + '/train/' + 'train_file_labels_' + str(ndim) + '.csv')
        label_out_path.parent.mkdir(parents=True, exist_ok=True)
        train_df = pd.DataFrame({'name': inp_names, 'Label': inp_labels})
        train_df.to_csv(label_out_path)
    
    malware_cfg_path = cfg_path + 'Test_CFG/Malware_CFG/'
    benign_cfg_path = cfg_path + 'Test_CFG/Benign_CFG/'
    
    Malware_graphs, Malware_names = loadCFG(malware_cfg_path, n_malware_test)
    Benign_graphs, Benign_names = loadCFG(benign_cfg_path, n_benign_test)
    
    n_percent_train = 0.0
    empty_graphs, inp2_graphs, empty_labels, inp2_labels, empty_num, n_inp2, empty_names, inp2_names = \
            train_test_divide(Malware_graphs, Benign_graphs, Malware_names, Benign_names, n_malware_test, n_benign_test,
                              n_percent_train)
    inp2_graphs, inp2_labels, inp2_names = DocShuffle(inp2_graphs, inp2_labels, inp2_names)     
    for ndim in ndims_list:
        print("Dimensions(test set): ", ndim)
        model = GL2Vec(dimensions=ndim)

        print("Fitting model")
        model.fit(vocab_train_graphs)
        print("Inferring graphs")
        X_vec = model.infer(inp2_graphs)
        vector_out_path = Path(model_path + '/test/' + 'test_file_vectors_' + str(ndim) + '.csv')
        vector_out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(X_vec).to_csv(vector_out_path, header=None, index=None)
        label_out_path = Path(model_path + '/test/' + 'test_file_labels_' + str(ndim) + '.csv')
        label_out_path.parent.mkdir(parents=True, exist_ok=True)
        train_df = pd.DataFrame({'name': inp2_names, 'Label': inp2_labels})
        train_df.to_csv(label_out_path)
    return model_path