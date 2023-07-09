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


def graph_embedding_training(cfg_path, output_path, ndims_list=[8]):
    malware_cfg_path = cfg_path + 'Malware_CFG/'
    benign_cfg_path = cfg_path + 'Benign_CFG/'

    n_precent_train = 0.2  # percentage for vocabulary  training (20% validation)
    
    # Load .gpickle CFG files
    Malware_graphs, Malware_names = loadCFG(malware_cfg_path, n_malware)
    Benign_graphs, Benign_names = loadCFG(benign_cfg_path, n_benign)

    ## Train divide for vocabulary training graphs
    vocab_train_graphs, train_graphs, vocab_train_labels, train_labels, n_vocab_train, n_train, vocab_train_names, train_names = \
        train_test_divide(Malware_graphs, Benign_graphs, Malware_names, Benign_names, n_malware, n_benign, n_precent_train)
    
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

    ########## Training Graph2Vec* Model ##############

    ## Parameters
    param = WL_parameters()
    param._set_seed()

    # Graph2Vec dimensions
    # ndims_list = [2, 4, 8, 16, 32, 64, 128, 256]
    # ndims_list = [2]

    # Train and save Graph2Vec model
    train_G2V_model(vocab_train_graphs, vocab_train_labels, vocab_train_names, param, ndims_list, save_model=True, output_path=output_path)

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
    print('Graph2Vec inference')
    # test_vector, test_vector_labels, test_vector_names = inferG2V(test_graphs, test_labels, test_names, param)

    # Create WL hash word documents for testing set
    print('Creating WL hash words for remaining training set')
    train_documents = createWLhash(train_graphs, param)

    if 'train_graphs' in os.environ:
        del train_graphs

    for ndim in ndims_list:
        print('Dimensions = ', ndim)

        ## Parameters
        param = WL_parameters(dimensions=ndim)
        param._set_seed()

        # model path
        model_path = output_path + 'd2v_models/'
        model_name = (model_path + '/' + 'd2v_model_' + str(param.dimensions) + '.model')

        try:
            d2v_model = Doc2Vec.load(model_name)
        except Exception as e:
            print("ERROR!!!!!!! : %s" % e)

        ## Shuffling of the data
        print('Shuffling data')
        train_corpus, train_vector_labels, train_vector_names = DocShuffle(train_documents, train_labels, train_names)

        ## Doc2Vec inference
        print('Doc2Vec inference')
        train_vector = np.array([d2v_model.infer_vector(d.words) for d in train_corpus])

        vector_out_path = Path(model_path + '/' + 'train_file_vectors_' + str(param.dimensions) + '.csv')
        vector_out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(train_vector).to_csv(vector_out_path, header=None, index=None)

        label_out_path = Path(model_path + '/' + 'train_file_labels_' + str(param.dimensions) + '.csv')
        label_out_path.parent.mkdir(parents=True, exist_ok=True)
        train_df = pd.DataFrame({'name': train_vector_names, 'Label': train_vector_labels})
        train_df.to_csv(label_out_path)

        ## Visualizing
        print('Visualizations')
        twoD_tsne_vector, fig = TSNE_2D_plot(train_vector, train_vector_labels, n_train, param.dimensions,
                                             return_plot=True)

        fig_name = model_path + '/' + 'train_vector_' + str(param.dimensions) + '-dims.png'
        fig.savefig(fig_name)
        plt.clf()

    return param, model_path

def clustering_training(output_path, cluster_alg_name =['Kmeans'], ndims_list = [8]):
    save_model = True
    # ndims_list = [2, 4, 8, 16, 32, 64, 128, 256]
    # ndims_list = [2]

    # cluster_alg_name = ['Kmeans', 'spectral', 'Aggloromative', 'DBSCAN']

    # cluster_alg_name = ['DBSCAN']
    
    for ndim in ndims_list:
        print('Dimensions = ', ndim)

        # load data
        model_path = output_path +  'd2v_models/'
        vector_path = model_path + '/' + 'train_file_vectors_' + str(ndim) + '.csv'
        label_path = model_path + '/' + 'train_file_labels_' + str(ndim) + '.csv'

        train_vector = pd.read_csv(vector_path, header=None).values

        vector = StandardScaler().fit_transform(train_vector)

        train_df = pd.read_csv(label_path)
        train_vector_labels = train_df['Label'].tolist()

        X_train, X_val, y_train, y_val = train_test_split(
            vector, train_vector_labels, test_size=0.4, random_state=0)

        ## Visualizing
        print('generating clustering and visualizations...')
        twoD_tsne_vector, fig = TSNE_2D_plot(X_train, y_train, len(y_train), ndim,
                                             return_plot=True)

        fig_name = model_path + '/' + 'val_vector_' + str(ndim) + '-dims.png'
        fig.savefig(fig_name)
        plt.clf()

        if 'fig' in os.environ:
            del os.environ['fig']

        for cluster_alg in cluster_alg_name:

            if cluster_alg == 'Kmeans':
                hyper_para_name = 'n_clusters'
                random_state = 0
                hyper_para_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
                hyper_para_list = np.arange(2, 31, step=1)
            elif cluster_alg == 'spectral':
                hyper_para_name = 'n_clusters'
                assign_labels = 'discretize'
                hyper_para_list = np.arange(2, 31, step=1)
            elif cluster_alg == 'Aggloromative':
                hyper_para_name = 'n_clusters'
                linkage = 'ward'
                hyper_para_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
                hyper_para_list = np.arange(2, 31, step=1)
            elif cluster_alg == 'DBSCAN':
                hyper_para_name = 'eps'
                min_samples = 1
                hyper_para_list = np.arange(5, 150, step=5)

            # hyper_para_name, hyper_para_list = get_clf_hyper_para(cluster_alg)

            print(cluster_alg)
            cluster_valuation = pd.DataFrame()

            for hyper_para in hyper_para_list:
                print('hyper parameter = ', hyper_para)
                if cluster_alg == 'Kmeans':
                    clustering_model, fig = generate_kmeans_clustering(X_train, y_train, twoD_tsne_vector,
                                                                       n_cluster=hyper_para, random_state=random_state)
                    y_pred = clustering_model.predict(X_val)
                elif cluster_alg == 'spectral':
                    clustering_model, fig = generate_spectral_clustering(X_train, y_train, twoD_tsne_vector,
                                                                         n_cluster=hyper_para,
                                                                         assign_labels=assign_labels)
                    y_pred = clustering_model.fit_predict(X_val)
                elif cluster_alg == 'Aggloromative':
                    clustering_model, fig = generate_AgglomerativeCluster(X_train, y_train, twoD_tsne_vector,
                                                                          n_cluster=hyper_para, linkage=linkage)
                    y_pred = clustering_model.fit_predict(X_val)
                elif cluster_alg == 'DBSCAN':
                    clustering_model, fig = generate_DBSCAN(X_train, y_train, hyper_para / 100, min_samples,
                                                            twoD_tsne_vector)
                    y_pred = clustering_model.fit_predict(X_val)

                # cluster_valuation.loc[0,len(cluster_valuation.index)] = cluster_evaluation(y_val, y_pred)
                eval, cntg = cluster_evaluation(y_val, y_pred)
                print(cntg)
                cluster_valuation = cluster_valuation.append( eval, ignore_index=True)
                cluster_valuation.reset_index()

                if save_model:
                    cluster_model_path = model_path + '/' + cluster_alg + '/cluster_models'

                    os.makedirs(cluster_model_path, exist_ok=True)

                    cluster_model_name = (cluster_model_path + '/' + 'clustering_model_' + str(
                        ndim) + '-dims_' + str(hyper_para) + '-clusters.sav')

                    pickle.dump(clustering_model, open(cluster_model_name, 'wb'))

                # fig_name = Path(model_path + '/'+ cluster_alg+'/validation/' + 'val_clustering_' + str(ndims) + '-dims_'+ str(n_cluster)+'-clusters.png')
                # fig_name.parent.mkdir(parents=True, exist_ok=True)
                # fig.savefig(fig_name)

            cluster_valuation.insert(0, hyper_para_name, hyper_para_list)

            valuation_name = Path(model_path + '/' + cluster_alg + '/validation/' + 'val_clustering_evaluation_' + str(
                ndim) + '-dims.csv')
            valuation_name.parent.mkdir(parents=True, exist_ok=True)
            cluster_valuation.to_csv(valuation_name)
    return 0

def cluster_prediction(cfg_path, output_path, param, cluster_alg_name =['Kmeans'], ndims_list = [8]):
    ##### Testing ######################

    # load testing data
    print('Load test CFG data')
    test_graphs, test_file_names, test_Labels, n_test = loadTestCFG(cfg_path, n_test_malware, n_test_benign)

    ## Create WL hash word documents for testing set
    print('Creating WL hash words for testing set')
    test_documents = createWLhash(test_graphs, param)

    for ndims in ndims_list:
        print('Dimensions = ', ndims)

        ## Parameters
        param = WL_parameters(dimensions=ndims)
        param._set_seed()

        # model path
        model_path = output_path +'d2v_models'
        model_name = (model_path + '/' + 'd2v_model_' + str(param.dimensions) + '.model')

        try:
            d2v_model = Doc2Vec.load(model_name)
        except Exception as e:
            print("ERROR - d2v model not found!!!!!!! : %s" % e)

        # Doc2Vec inference
        print('Doc2Vec inference')
        test_vector = np.array([d2v_model.infer_vector(d.words) for d in test_documents])

        vector_out_path = Path(model_path + '/Test/' + 'test_file_vectors_' +str(param.dimensions) + '.csv')
        vector_out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(test_vector).to_csv(vector_out_path, header=None, index= None)

        label_out_path = Path(model_path+'/Test/'+'file_labels_'+str(param.dimensions)+'.csv')
        label_out_path.parent.mkdir(parents=True, exist_ok=True)
        test_df = pd.DataFrame({'name': test_file_names, 'Label': test_Labels})
        test_df.to_csv(label_out_path)

        # Visualizing
        print('Visualizations')
        twoD_tsne_vector, fig =TSNE_2D_plot(test_vector, test_Labels, n_test, param.dimensions, return_plot=True)

        fig_name = model_path + '/Test/' + 'test_vector_' + str(param.dimensions)  + '-dims.png'
        fig.savefig(fig_name)
        plt.clf()

        for cluster_alg in cluster_alg_name:
            print(cluster_alg)
            cluster_valuation = pd.DataFrame()

            if cluster_alg == 'Kmeans':
                hyper_para_name = 'n_clusters'
                random_state = 0
                hyper_para_list = [ 2, 3, 4, 5, 6, 7, 8, 9, 10, 15,  20]
            elif cluster_alg == 'spectral':
                hyper_para_name = 'n_clusters'
                assign_labels = 'discretize'
                hyper_para_list = np.arange(2, 31, step = 1)
            elif cluster_alg == 'Aggloromative':
                hyper_para_name = 'n_clusters'
                linkage = 'ward'
                hyper_para_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
            elif cluster_alg == 'DBSCAN':
                hyper_para_name = 'eps'
                min_samples = 2
                hyper_para_list = np.arange(5,150 , step = 5)

            hyper_para_name, hyper_para_list = get_clf_hyper_para(cluster_alg)

            for hyper_para in hyper_para_list:
                print('hyper parameter = ', hyper_para)
                cluster_model_path = model_path + '/' + cluster_alg + '/cluster_models'
                load_model=True
                if load_model:

                    os.makedirs(cluster_model_path, exist_ok=True)

                    cluster_model_name = (cluster_model_path + '/' + 'clustering_model_' + str(
                        ndims) + '-dims_' + str(hyper_para) + '-clusters.sav')

                    # load the model from disk
                    try:
                        clustering_model = pickle.load(open(cluster_model_name, 'rb'))
                    except Exception as e:
                        print("ERROR - clustering model not found!!!!!!! : %s" % e)

                if cluster_alg == 'Kmeans':
                    array_float = np.array(test_vector, dtype=np.float64)
                    y_pred = clustering_model.predict(array_float)
                elif cluster_alg == 'spectral':
                    y_pred = clustering_model.fit_predict(test_vector)
                elif cluster_alg == 'Aggloromative':
                    y_pred = clustering_model.fit_predict(test_vector)
                elif cluster_alg == 'DBSCAN':
                    y_pred = clustering_model.fit_predict(test_vector)

                # cluster_valuation.loc[0,len(cluster_valuation.index)] = cluster_evaluation(y_val, y_pred)

                eval, cntg = cluster_evaluation(test_Labels, y_pred)
                print(cntg)
                print(eval)
                cluster_valuation = cluster_valuation.append( eval, ignore_index= True)
                cluster_valuation.reset_index()

                predict_out_path = Path(model_path + '/' + cluster_alg + '/Test/' + 'file_predictions_' + str(ndims) + '-dims_'+ str(hyper_para) + '-clusters.csv')
                predict_out_path.parent.mkdir(parents=True, exist_ok=True)
                test_df = pd.DataFrame({'name': test_df['name'].tolist(), 'Label': test_Labels, 'Predict': y_pred})
                test_df.to_csv(predict_out_path)

                fig = plot_clusters(twoD_tsne_vector, y_pred, alg_name= cluster_alg, hyper_para_name=hyper_para_name,hyp_para=hyper_para, ndims=ndims)

                fig_name = Path(model_path + '/' + cluster_alg+'/Test/' + 'test_clustering_' + str(ndims) + '-dims_'+
                                str(hyper_para)+'-clusters.png')
                fig_name.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(fig_name)
                plt.clf()

            cluster_valuation.insert(0, hyper_para_name, hyper_para_list)

            valuation_name = Path(model_path + '/' + cluster_alg+'/Test/' + 'test_clustering_evaluation_' + str(ndims) +'-dims.csv')
            valuation_name.parent.mkdir(parents=True, exist_ok=True)
            cluster_valuation.to_csv(valuation_name)


def get_clf_hyper_para(cluster_alg):

    if cluster_alg == 'Kmeans':
        hyper_para_name = 'n_clusters'
        random_state = 0
        hyper_para_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
    elif cluster_alg == 'spectral':
        hyper_para_name = 'n_clusters'
        assign_labels = 'discretize'
        hyper_para_list = np.arange(2, 31, step = 1)
    elif cluster_alg == 'Aggloromative':
        hyper_para_name = 'n_clusters'
        linkage = 'ward'
        hyper_para_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
    elif cluster_alg == 'DBSCAN':
        hyper_para_name = 'eps'
        min_samples = 2
        hyper_para_list = np.arange(5,150 , step = 5)
    return hyper_para_name, hyper_para_list
'''

def supervised_learning(output_path, superv_alg_name, ndims_list, emb, clf_type='plain'):
    for ndim in ndims_list:
        print('Dimensions = ', ndim)

    # load data
        #model_path = output_path + 'd2v_models/'
        model_path = output_path + emb + '_models/'
        train_vector_path = model_path + 'train/train_file_vectors_' + str(ndim) + '.csv'
        train_label_path = model_path + 'train/train_file_labels_' + str(ndim) + '.csv'

        test_vector_path = model_path + 'test/test_file_vectors_' + str(ndim) + '.csv'
        test_label_path = model_path + 'test/test_file_labels_' + str(ndim) + '.csv'
        
        train_vector = pd.read_csv(train_vector_path, header=None).values
        X_train = StandardScaler().fit_transform(train_vector)
        
        test_vector = pd.read_csv(test_vector_path, header=None).values
        X_test = StandardScaler().fit_transform(test_vector)
        
        train_df = pd.read_csv(train_label_path)
        y_train = train_df['Label'].tolist() 

        train_df = pd.read_csv(test_label_path)
        y_test = train_df['Label'].tolist() 
        
        # Convert 'Benign/Malware' labels to binary (Benign->0 and Malware->1)
        y_train_bin = [1 if idx == 'Malware' else 0 for idx in y_train]
        y_test_bin = [1 if idx == 'Malware' else 0 for idx in y_test]
        
        #print(len(X_train), len(X_test))
        #print(len(y_train_bin), len(y_test_bin))
        
        X_train_size = len(X_train)
        y_train_size = len(y_train)
        if clf_type == 'optimized':
            for alg in superv_alg_name:
                print("Current Alg: ", alg)
                model, optimizing_record, training_time = supervised_learning_caller_optimized(alg, X_train, y_train_bin, 
                                                                                               X_test, y_test_bin)
                output_path1, model = supervised_methods_evaluation(alg, model, X_train, y_train_bin, X_test, y_test_bin,
                                              ndim, X_train_size, y_train_size,
                                              output_path, emb, training_time,
                                              optimized=True, optimizing_tuple=optimizing_record)
                #group_plotter(superv_alg_name, emb, ndims_list, output_path, optimized=True)
        elif clf_type == 'plain':
            for alg in superv_alg_name:
                model, training_time = plain_clf_runner(alg, X_train, y_train_bin)
                output_path1, model = supervised_methods_evaluation(alg, model, X_train, y_train_bin, X_test, y_test_bin,
                                              ndim, X_train_size, y_train_size,
                                              output_path, emb, training_time)
                #group_plotter(superv_alg_name, emb, ndims_list, output_path, optimized=False)
    return 0
'''

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
                                                                                               X_test, y_test_bin, True )

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
    ndims_list = [2, 4, 8, 16, 32, 64, 128, 256]
    
    #ndims_list = [ 512, 1024]
    
    #ndims_list = [ 2048 ]
    
    #ndims_list = [ 512, 1024, 2048, 4096 ]
    
    #ndims_list = [2, 4, 8]
    
    #ndims_list = [ 2, 4, 8, 16, 32, 64 ]
    
    # The hyper parameters for embedding is inside the function
    
    #embed_list = [ 'wlksvd' ] #,  'gl2v', 'd2v']
    
    embed_list = [ 'd2v' ] #, 'wlksvd' ]
    
    #model_path = create_embedding(embed_list, cfg_path, output_path, ndims_list, n_malware, n_benign, n_test_malware, n_test_benign, isTrain = True)
    
    ####### 3. Unsupervised clustering algorithm training with hold-out validation. ##################
    print('******************* STEP: 3 *******************')
    # Unsupervised clustering algorithms
    #cluster_alg_name = ['Kmeans', 'spectral', 'Aggloromative', 'DBSCAN']
    #cluster_alg_name = ['Kmeans', 'Aggloromative']
    # The hyper parameters for each clustering method is inside the function

    #clustering_training(output_path, cluster_alg_name, ndims_list)

    #Supervised Classification methods
    
    #all implemented classifiers, mnb, compnb, catnb, and mlp are not functional
    #superv_alg_name = [ 'knn', 'rnn', 'lsvc', 'nsvc', 'svc', 'gp', 'dt', 'rf', 'ada', 'gnb', 'mnb', 
    #                    'compnb', 'catnb' ,'qda', 'lda', 'mlp']
    
    
    
    #superv_alg_name = [  'ada', 'dt' ]
    
    #superv_alg_name = [  'nsvc', 'rf']
    
    #superv_alg_name = [ 'gnb', 'mnb', 'compnb']
    
    superv_alg_name = [ 'gp' , 'knn', 'svc']
    
    #superv_alg_name = [ 'svc','lda' ] #, 'qda']
    
    #superv_alg_name = [ 'ada', 'dt', 'nsvc', 'svc', 'lda', 'qda' ]   
    
    #superv_alg_name = [ 'ridge', 'pa', 'sgd', 'perc', 'lsvc', 'rf' ]
    
    #superv_alg_name = ['lsvc']
    
    clf_type1 = 'plain'
    #clf_type2 = 'optimized'

    #embed_list = [ 'wlksvd' ]
    
    for emb in embed_list:
        
        print("Running Classifiers on " + classifier_name_expand(emb) + " embedded vectors" )
    
        print("Plain classifier check")
        supervised_learning(output_path, superv_alg_name, ndims_list, emb, clf_type1)
    
        #print("Optimized classifier check")
        #supervised_learning(output_path, superv_alg_name, ndims_list, emb, clf_type2)
        
    print("Running Plotting Functions")
    for emb in embed_list:

        #group_plotter_by_dim(superv_alg_name, emb, ndims_list, output_path, 'optimized')
        group_plotter_by_dim(superv_alg_name, emb, ndims_list, output_path, 'plain')
        
        #group_plotter_by_alg(superv_alg_name, emb, ndims_list, output_path, 'optimized')
        group_plotter_by_alg(superv_alg_name, emb, ndims_list, output_path, 'plain')
       
    
    ####### 4. Cluster prediction for Test dataset ##################
    print('******************* STEP: 4 *******************')
    #cluster_prediction(test_cfg_path, output_path, param, cluster_alg_name, ndims_list)

    
    
    
    print('******************* Process Finished *******************')
