from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.manifold import TSNE
from sklearn.cluster import *
from sklearn import metrics
from sklearn.metrics.cluster import contingency_matrix

from sklearn.model_selection import train_test_split

from g2v_util import *

import pickle


def clustering_training(output_path, emb, cluster_alg_name, ndims_list):
    save_model = True
    
    if emb == 'wlksvd':
        model_path = output_path + 'wlksvd_models/'
    elif emb == 'd2v':
        model_path = output_path + 'd2v_models/'
    
    for ndim in ndims_list:
        print('Dimensions = ', ndim)

        # load data
        
        vector_path = model_path + '/train/' + 'train_file_vectors_' + str(ndim) + '.csv'
        label_path = model_path + '/train/' + 'train_file_labels_' + str(ndim) + '.csv'

        train_vector = pd.read_csv(vector_path, header=None).values

        # vector = StandardScaler().fit_transform(train_vector)
        scaler = MaxAbsScaler()
        scaler.fit(train_vector)
        vector = scaler.transform(train_vector)

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

                clustering_model.scaler = scaler

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

def cluster_prediction(cfg_path, output_path, cluster_alg_name , emb, ndims_list, n_test_malware, n_test_benign):
    for ndim in ndims_list:
        if emb == 'wlksvd':
            model_path = output_path +'wlksvd_models'
            model_name = (model_path + '/' + 'wlksvd_model_' + str(ndim) + '.model')
            
        elif emb == 'd2v':
            model_path = output_path +'d2v_models'
            model_name = (model_path + '/' + 'd2v_model_' + str(ndim) + '.model')
        
        vector_path = model_path + '/test/' + 'test_file_vectors_' + str(ndim) + '.csv'
        label_path = model_path + '/test/' + 'test_file_labels_' + str(ndim) + '.csv'

        test_vector = pd.read_csv(vector_path, header=None).values

        test_df = pd.read_csv(label_path)
        y_test = test_df['Label'].tolist()
            
                ## Visualizing
        print('generating clustering and visualizations...')
        twoD_tsne_vector, fig = TSNE_2D_plot(test_vector, y_test, len(y_test), ndim,
                                             return_plot=True)

        fig_name = model_path + '/test/' + 'test_vector_' + str(ndim) + '-dims.png'
        fig.savefig(fig_name)
        plt.clf()    
            
        for cluster_alg in cluster_alg_name:
            print(cluster_alg)
            cluster_valuation = pd.DataFrame()

            # if cluster_alg == 'Kmeans':
            #     hyper_para_name = 'n_clusters'
            #     random_state = 0
            #     hyper_para_list = [ 2, 3, 4, 5, 6, 7, 8, 9, 10, 15,  20]
            # elif cluster_alg == 'spectral':
            #     hyper_para_name = 'n_clusters'
            #     assign_labels = 'discretize'
            #     hyper_para_list = np.arange(2, 31, step = 1)
            # elif cluster_alg == 'Aggloromative':
            #     hyper_para_name = 'n_clusters'
            #     linkage = 'ward'
            #     hyper_para_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
            # elif cluster_alg == 'DBSCAN':
            #     hyper_para_name = 'eps'
            #     min_samples = 2
            #     hyper_para_list = np.arange(5,150 , step = 5)

            hyper_para_name, hyper_para_list = get_clf_hyper_para(cluster_alg)

            for hyper_para in hyper_para_list:
                print('hyper parameter = ', hyper_para)
                cluster_model_path = model_path + '/' + cluster_alg + '/cluster_models'
                load_model=True
                if load_model:

                    os.makedirs(cluster_model_path, exist_ok=True)

                    cluster_model_name = (cluster_model_path + '/' + 'clustering_model_' + str(ndim) + '-dims_' + str(hyper_para) + '-clusters.sav')

                    # load the model from disk
                    try:
                        clustering_model = pickle.load(open(cluster_model_name, 'rb'))
                    except Exception as e:
                        print("ERROR - clustering model not found!!!!!!! : %s" % e)

                scaler = clustering_model.scaler

                X_test = scaler.transform(test_vector)

                print(X_test.shape)
                if cluster_alg == 'Kmeans':
                    array_float = np.array(X_test, dtype=np.float64)
                    print(array_float.shape)
                    y_pred = clustering_model.predict(array_float)
                elif cluster_alg == 'spectral':
                    y_pred = clustering_model.fit_predict(X_test)
                elif cluster_alg == 'Aggloromative':
                    y_pred = clustering_model.fit_predict(X_test)
                elif cluster_alg == 'DBSCAN':
                    y_pred = clustering_model.fit_predict(X_test)

                #cluster_valuation.loc[0,len(cluster_valuation.index)] = cluster_evaluation(y_val, y_pred)

                eval, cntg = cluster_evaluation(y_test, y_pred)
                print(cntg)
                print(eval)
                cluster_valuation = cluster_valuation.append(eval, ignore_index= True)
                cluster_valuation.reset_index()

                predict_out_path = Path(model_path + '/' + cluster_alg + '/test/' + 'file_predictions_' + str(ndim) + '-dims_'+ str(hyper_para) + '-clusters.csv')
                predict_out_path.parent.mkdir(parents=True, exist_ok=True)
                test_df = pd.DataFrame({'name': test_df['name'].tolist(), 'Label': y_test, 'Predict': y_pred})
                test_df.to_csv(predict_out_path)

                fig = plot_clusters(twoD_tsne_vector, y_pred, alg_name= cluster_alg, hyper_para_name=hyper_para_name,hyp_para=hyper_para, ndims=ndim)

                fig_name = Path(model_path + '/' + cluster_alg + '/test/' + 'test_clustering_' + str(ndim) + '-dims_'+
                                str(hyper_para)+'-clusters.png')
                fig_name.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(fig_name)
                plt.clf()

            cluster_valuation.insert(0, hyper_para_name, hyper_para_list)

            valuation_name = Path(model_path + '/' + cluster_alg + '/test/' + 'test_clustering_evaluation_' + str(ndim) +'-dims.csv')
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


def TSNE_2D_plot(vector, labels, n_vec, dimensions, return_plot = False):
    twoD_embedded_graphs = TSNE(n_components=2).fit_transform(vector)

    idx_malware = [i for i in range(n_vec) if labels[i] == 'Malware']
    idx_benign = [i for i in range(n_vec) if labels[i] == 'Benign']


    # plt.subplot(1,2,1)
    plt.plot(twoD_embedded_graphs[idx_malware, 0], twoD_embedded_graphs[idx_malware, 1], 'ro', label='malware', alpha =0.4)
    plt.legend(loc='upper left')
    # plt.subplot(1,2,2)
    plt.plot(twoD_embedded_graphs[idx_benign, 0], twoD_embedded_graphs[idx_benign, 1], 'bo', label='benign', alpha= 0.4)
    plt.legend(loc='upper left')
    plt.suptitle('Graph2Vec (' + str(dimensions) + ' dims) \n TSNE visualization of input graphs')
    # plt.legend(bbox_to_anchor=(1.05, 1))

    fig1 = plt.gcf()
    plt.show()

    if return_plot:
        return twoD_embedded_graphs, fig1
    else:
        plt.clf()
        return twoD_embedded_graphs



def generate_DBSCAN(vector, labels, eps, min_samples, twoD_vector):
    _, ndim = vector.shape

    # Get results for DBSCAN clustering algorithm
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(vector)
    # Get DBSCAN cluster labels
    cluster_DBSCAN_Labels = clustering.labels_

    core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
    core_samples_mask[clustering.core_sample_indices_] = True

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(cluster_DBSCAN_Labels)) - (1 if -1 in cluster_DBSCAN_Labels else 0)
    n_noise_ = list(cluster_DBSCAN_Labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, cluster_DBSCAN_Labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels, cluster_DBSCAN_Labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels, cluster_DBSCAN_Labels))
    print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels, cluster_DBSCAN_Labels))
    print(
        "Adjusted Mutual Information: %0.3f"
        % metrics.adjusted_mutual_info_score(labels, cluster_DBSCAN_Labels)
    )
    # print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(vector, cluster_DBSCAN_Labels))
    # confusion_matrix()

    # #############################################################################
    # Plot result
    import matplotlib.pyplot as plt

    # Black removed and is used for noise instead.
    unique_labels = set(cluster_DBSCAN_Labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = cluster_DBSCAN_Labels == k

        xy = twoD_vector[class_member_mask ]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )

    plt.title('DBSCAN with dim ( ' + str(ndim) + ')\nEstimated number of clusters: ' + str(n_clusters_)
              + '\n eps = '+str(eps))
    fig = plt.gcf()
    # plt.show()
    # fname = './Dikedataset_graphs/figs/synthetic-graph-comparison-graph2vec-' + str(ndims) + '-dims.png'
    # fname = './SSD_data_test/SSD_graphs/SSD_DBSCAN-' + str(ndim) + '-dims.png'
    # fig1.savefig(fname)
    plt.clf()
    return clustering, fig

def generate_AgglomerativeCluster(vector, labels, twoD_vector, n_cluster =2 , linkage = 'ward' ):

    _, ndim = vector.shape
    clustering = AgglomerativeClustering(n_clusters= n_cluster, linkage= linkage).fit(vector)

    # Get cluster labels
    cluster_Labels = clustering.labels_

    core_samples_mask = np.zeros_like(cluster_Labels, dtype=bool)
    # core_samples_mask[clustering.core_sample_indices_] = True

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(cluster_Labels)) - (1 if -1 in cluster_Labels else 0)
    n_noise_ = list(cluster_Labels).count(-1)
    # Plot result
    import matplotlib.pyplot as plt

    # Black removed and is used for noise instead.
    unique_labels = set(cluster_Labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = cluster_Labels == k

        xy = twoD_vector[class_member_mask ]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=7,
        )


    plt.title('Agglomerative with dim ( ' + str(ndim) + ')\nEstimated number of clusters: ' + str(n_clusters_)
              + '\n #clusters = ' + str(n_cluster) + ', linkage = ' + linkage)
    fig = plt.gcf()
    # plt.show()
    # fname = './Dikedataset_graphs/figs/synthetic-graph-comparison-graph2vec-' + str(ndims) + '-dims.png'
    # fname = './SSD_data_test/SSD_graphs/SSD_agglomerative-' + str(ndim) + '-dims.png'
    # fig1.savefig(fname)
    plt.clf()
    return clustering, fig

def generate_spectral_clustering(vector, labels, twoD_vector, n_cluster = 2, assign_labels = 'discretize' ):
    _, ndim = vector.shape
    clustering = SpectralClustering(n_clusters= n_cluster, assign_labels = assign_labels, random_state=0).fit(vector)

    # Get cluster labels
    cluster_Labels = clustering.labels_

    core_samples_mask = np.zeros_like(cluster_Labels, dtype=bool)
    # core_samples_mask[clustering.core_sample_indices_] = True

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(cluster_Labels)) - (1 if -1 in cluster_Labels else 0)
    n_noise_ = list(cluster_Labels).count(-1)
    # Plot result


    # Black removed and is used for noise instead.
    unique_labels = set(cluster_Labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = cluster_Labels == k

        xy = twoD_vector[class_member_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=5,
        )

    plt.title('Spectral with dim ( ' + str(ndim) + ')\nEstimated number of clusters: ' + str(n_clusters_)
              + '\n #clusters = ' + str(n_cluster) + ', labels = ' + assign_labels)
    fig = plt.gcf()
    # plt.show()
    # fname = './Dikedataset_graphs/figs/synthetic-graph-comparison-graph2vec-' + str(ndims) + '-dims.png'
    # fname = './SSD_data_test/SSD_graphs/SSD_agglomerative-' + str(ndims) + '-dims.png'
    # fig.savefig(fname)
    plt.clf()

    return clustering, fig

def generate_kmeans_clustering(vector, labels, twoD_vector, n_cluster = 2, random_state = 0):
    _, ndim = vector.shape
    clustering = KMeans(n_clusters=n_cluster, random_state = random_state).fit(vector)

    # Get cluster labels
    cluster_Labels = clustering.labels_

    core_samples_mask = np.zeros_like(cluster_Labels, dtype=bool)
    # core_samples_mask[clustering.core_sample_indices_] = True

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(cluster_Labels)) - (1 if -1 in cluster_Labels else 0)
    n_noise_ = list(cluster_Labels).count(-1)
    # Plot result

    # Black removed and is used for noise instead.
    unique_labels = set(cluster_Labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    labels = {}

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = cluster_Labels == k

        xy = twoD_vector[class_member_mask]

        labels[str(k + 1)] = xy[0]

        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=5,
            alpha=0.4,
        )

    for each in labels.keys():
        plt.annotate(each, labels[each], weight='bold', size=20)


    plt.title('Kmeans with dim ( ' + str(ndim) + ')\n #clusters = ' + str(n_cluster) )
    fig = plt.gcf()
    plt.show()
    # fname = './Dikedataset_graphs/figs/synthetic-graph-comparison-graph2vec-' + str(ndims) + '-dims.png'
    # fname = './SSD_data_test/SSD_graphs/SSD_agglomerative-' + str(ndims) + '-dims.png'
    # fig.savefig(fname)
    plt.clf()

    return clustering, fig

def cluster_evaluation(labels_true, labels_pred, algorithm= ''):

    ri = metrics.rand_score(labels_true, labels_pred)   # RAND score
    ari = metrics.adjusted_rand_score(labels_true, labels_pred) # Adjusted RAND score

    mis = metrics.mutual_info_score(labels_true, labels_pred)  # mutual info score
    amis = metrics.adjusted_mutual_info_score(labels_true, labels_pred)    # adjusted mutual information score
    nmis = metrics.normalized_mutual_info_score(labels_true, labels_pred)  # normalized mutual info score

    hmg = metrics.homogeneity_score(labels_true, labels_pred)   # homogeneity
    cmplt = metrics.completeness_score(labels_true, labels_pred)    # completeness
    v_meas = metrics.v_measure_score(labels_true, labels_pred)   # v_measure score

    fowlkes_mallows = metrics.fowlkes_mallows_score(labels_true, labels_pred)   # Fowlkes-Mallows scores

    cntg_mtx = contingency_matrix(labels_true, labels_pred)     # Contingency Matrix

    d = {'RAND' : ri , 'ARAND': ari, 'MIS' : mis, 'AMIS' : amis, 'NMIS' : nmis, 'Hmg' : hmg, 'Cmplt' : cmplt,
                 'V_meas' : v_meas, 'FMs' : fowlkes_mallows}
    # df = pd.DataFrame(data =  d)
    return d, cntg_mtx

def model_evaluation(X, labels):

    silhoutte = metrics.silhouette_score(X, labels, metric='euclidean')     # Silhouette Coefficient
    calinski_harabasz = metrics.calinski_harabasz_score(X, labels)  # Calinski-Harabasz Index
    davies_bouldin = metrics.davies_bouldin_score(X, labels)    # Davies-Bouldin Index

    d = {'Silhoutte' : silhoutte, 'Calinski_Harbasz' : calinski_harabasz, 'Davies_Bouldin' : davies_bouldin}
    # df = pd.DataFrame(d, index=)
    return d

def plot_clusters(twoD_vector, cluster_Labels, alg_name ='', hyper_para_name='default', hyp_para= 0, ndims='default'):
    core_samples_mask = np.zeros_like(cluster_Labels, dtype=bool)
    # core_samples_mask[clustering.core_sample_indices_] = True

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(cluster_Labels)) - (1 if -1 in cluster_Labels else 0)
    n_noise_ = list(cluster_Labels).count(-1)
    # Plot result

    # Black removed and is used for noise instead.
    unique_labels = set(cluster_Labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    labels = {}

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = cluster_Labels == k

        xy = twoD_vector[class_member_mask]

        labels[str(k+1)] = xy[0]

        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=5,
            alpha= 0.4,
        )

    for each in labels.keys():
         plt.annotate(each,labels[each], weight= 'bold', size = 20)

    plt.title( alg_name + ' with dim ( ' + str(ndims) + ')\n #'+ hyper_para_name +' = ' + str(hyp_para) )
    fig = plt.gcf()
    plt.show()
    # fname = './Dikedataset_graphs/figs/synthetic-graph-comparison-graph2vec-' + str(ndims) + '-dims.png'
    # fname = './SSD_data_test/SSD_graphs/SSD_agglomerative-' + str(ndims) + '-dims.png'
    # fig.savefig(fname)
    # plt.clf()
    return fig