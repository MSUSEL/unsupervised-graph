from sklearn.model_selection import train_test_split

from util import *
from cluster_util import *
import gc
import os

def create_CFG_datastet(prog_path, cfg_path, n_malware, n_benign):
    malware_binary_path = prog_path + 'malware/'
    benign_binary_path = prog_path + 'benign/'

    malware_cfg_path = cfg_path + 'Malware_CFG/'
    benign_cfg_path = cfg_path + 'Benign_CFG/'

    createSaveCFG(malware_binary_path, malware_cfg_path, n_malware)
    createSaveCFG(benign_binary_path, benign_cfg_path, n_benign)

def graph_embedding_training(cfg_path, output_path):
    malware_cfg_path = cfg_path + 'Malware_CFG/'
    benign_cfg_path = cfg_path + 'Benign_CFG/'

    n_precent_train = 0.2  # percentage for vocabulary  training (20% validation)

    # Load .gpickle CFG files
    Malware_graphs, Malware_names = loadCFG(malware_cfg_path)
    Benign_graphs, Benign_names = loadCFG(benign_cfg_path)

    ## Train divide for vocabulary training graphs
    vocab_train_graphs, train_graphs, vocab_train_labels, train_labels, n_vocab_train, n_train, vocab_train_names, train_names = \
        train_test_divide(Malware_graphs, Benign_graphs, Malware_names, Benign_names, n_malware, n_benign,
                          n_precent_train)

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
    ndims_list = [2, 4, 8, 16, 32, 64, 128, 256]
    ndims_list = [2]
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

    ## Graph2Vec* inference
    print('Graph2Vec inference')
    # test_vector, test_vector_labels, test_vector_names = inferG2V(test_graphs, test_labels, test_names, param)

    ## Create WL hash word documents for testing set
    print('Creating WL hash words for remaining training set')
    train_documents = createWLhash(train_graphs, param)

    if 'train_graphs' in os.environ:
        del (train_graphs)

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

def cluster_prediction(cfg_path, output_path):

    ##### Testing ######################

    # load testing data

    test_graphs, test_file_names, test_Labels, n_test = loadTestCFG(cfg_path)

    ## Parameters
    param = WL_parameters()
    param._set_seed()

    ndims_list = [2, 4, 8, 16, 32, 64, 128, 256]
    ndims_list = [2]
    ## Create WL hash word documents for testing set
    print('Creating WL hash words for testing set')
    test_documents = createWLhash(test_graphs, param)


    for ndims in ndims_list:
        print('Dimensions = ', ndims)

        ## Parameters
        param = WL_parameters(dimensions=ndims)
        param._set_seed()


        # model path
        model_path = output_path + 'd2v_models/'
        model_name = (model_path + '/' + 'd2v_model_' + str(param.dimensions) + '.model')

        try:
            d2v_model = Doc2Vec.load(model_name)
        except Exception as e:
            print("ERROR!!!!!!! : %s" % e)



        ## Doc2Vec inference
        print('Doc2Vec inference')
        test_vector = np.array([d2v_model.infer_vector(d.words) for d in test_documents])

        vector_out_path = Path(model_path + '/Test/' + 'file_vectors_' +str(param.dimensions) + '.csv')
        vector_out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(test_vector).to_csv(vector_out_path, header=None, index= None)

        label_out_path = Path(model_path+'/Test/'+'file_labels_'+str(param.dimensions)+'.csv')
        label_out_path.parent.mkdir(parents=True, exist_ok=True)
        test_df = pd.DataFrame({'name': test_file_names, 'Label': test_Labels})
        test_df.to_csv(label_out_path)


        ## Visualizing
        print('Visualizations')
        twoD_tsne_vector, fig =TSNE_2D_plot(test_vector, test_Labels, n_test, param.dimensions, return_plot=True)

        fig_name = model_path + '/Test/' + 'test_vector_' + str(param.dimensions)  + '-dims.png'
        fig.savefig(fig_name)
        plt.clf()
    return 0

def clustering_training(output_path):
    save_model = True
    ndims_list = [2, 4, 8, 16, 32, 64, 128, 256]
    cluster_alg_name = ['Kmeans', 'spectral', 'Aggloromative', 'DBSCAN']

    cluster_alg_name = ['DBSCAN']

    ndims_list = [2]
    for ndim in ndims_list:
        print('Dimensions = ', ndim)

        # load data
        model_path = output_path +  'd2v_models/'
        vector_path = model_path + '/' + 'train_file_vectors_' + str(ndim) + '.csv'
        label_path = model_path + '/' + 'train_file_labels_' + str(ndim) + '.csv'

        test_vector = pd.read_csv(vector_path, header=None).values

        vector = StandardScaler().fit_transform(test_vector)

        train_df = pd.read_csv(label_path)
        train_vector_labels = train_df['Label'].tolist()

        X_train, X_val, y_train, y_val = train_test_split(
            vector, train_vector_labels, test_size=0.4, random_state=0)

        ## Visualizing
        print('generating clustering and visualizations...')
        twoD_tsne_vector, fig = TSNE_2D_plot(X_train, y_train, len(y_train), ndim,
                                             return_plot=True)

        fig_name = model_path + '/' + 'val_vector_' + str(ndim) + '-dims.png'
        # fig.savefig(fig_name)
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

            # hyper_para_list = [30]
            ## K-means
            print(cluster_alg)
            cluster_valuation = pd.DataFrame()

            for hyper_para in hyper_para_list:

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
                cluster_valuation = cluster_valuation.append(eval, ignore_index=True)
                cluster_valuation.reset_index()

                if save_model:
                    cluster_model_path = model_path + '/' + cluster_alg + '/validation'

                    os.makedirs(cluster_model_path, exist_ok=True)

                    cluster_model_name = (cluster_model_path + '/' + 'clustering_model' + str(
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

def cluster_prediction(cfg_path, output_path):
    ##### Testing ######################

    # load testing data

    test_graphs, test_file_names, test_Labels, n_test = loadTestCFG(cfg_path)

    ## Parameters
    param = WL_parameters()
    param._set_seed()

    # ndims_list = [2, 4, 8, 16, 32, 64, 128, 256]
    ndims_list = [2]
    ## Create WL hash word documents for testing set
    print('Creating WL hash words for testing set')
    test_documents = createWLhash(test_graphs, param)


    for ndims in ndims_list:
        print('Dimensions = ', ndims)

        ## Parameters
        param = WL_parameters(dimensions=ndims)
        param._set_seed()


        # model path
        model_path = output_path +'d2v_models/'
        model_name = (model_path + '/' + 'd2v_model_' + str(param.dimensions) + '.model')

        try:
            d2v_model = Doc2Vec.load(model_name)
        except Exception as e:
            print("ERROR!!!!!!! : %s" % e)



        ## Doc2Vec inference
        print('Doc2Vec inference')
        test_vector = np.array([d2v_model.infer_vector(d.words) for d in test_documents])

        vector_out_path = Path(model_path + '/Test/' + 'test_file_vectors_' +str(param.dimensions) + '.csv')
        vector_out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(test_vector).to_csv(vector_out_path, header=None, index= None)

        label_out_path = Path(model_path+'/Test/'+'file_labels_'+str(param.dimensions)+'.csv')
        label_out_path.parent.mkdir(parents=True, exist_ok=True)
        test_df = pd.DataFrame({'name': test_file_names, 'Label': test_Labels})
        test_df.to_csv(label_out_path)


        ## Visualizing
        print('Visualizations')
        twoD_tsne_vector, fig =TSNE_2D_plot(test_vector, test_Labels, n_test, param.dimensions, return_plot=True)

        fig_name = model_path + '/Test/' + 'test_vector_' + str(param.dimensions)  + '-dims.png'
        fig.savefig(fig_name)
        plt.clf()

if __name__ == "__main__":
    print('Binary file analysis using Graph2Vec and unsupervised clustering')

    # define paths for the binary files, CFG files, and Output files
    prog_path = './Binary_data/'
    cfg_path = './CFG_data/CFG_dataset/'
    output_path = './CFG_data/results/'
    info_path = './CFG_data/class_labels_no_dupes.csv'

    ######## 1. Creating CFGs from binary files ############
    print('STEP: 1')
    # Skip this step if you already have the CFG dataset
    # Note: The binary files are not provided

    # # Training binary files
    train_prog_path = prog_path + 'Train/'
    train_cfg_path = cfg_path + 'Train_CFG/'
    n_malware = 3000
    n_benign = 3000
    # create_CFG_datastet(train_prog_path, cfg_path, n_malware, n_benign)
    #
    # # Testing binary files
    test_prog_path = prog_path + 'Test/'
    test_cfg_path = cfg_path + 'Test_CFG/'
    n_malware = 1000
    n_benign = 1000
    # create_CFG_datastet(test_prog_path, cfg_path, n_malware, n_benign)

    ######## 2. Create vector representation for CFGs using 'Graph2Vec' graph embedding. ############
    print('STEP: 2')
    param, model_path = graph_embedding_training(train_cfg_path, output_path)

    ####### 3. Unsupervised clustering algorithm training with hold out validation. ##################
    print('STEP: 3')
    clustering_training(output_path)

    ####### 4. Cluster prediction for Test dataset ##################
    print('STEP: 4')
    cluster_prediction(cfg_path, output_path)


