from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans

from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster import contingency_matrix
from sklearn import metrics

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.manifold import TSNE

import pickle



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
            alpha =0.4
        )

    plt.title('DBSCAN with dim ( ' + str(ndim) + ')\nEstimated number of clusters: ' + str(n_clusters_)
              + '\n eps = '+str(eps))
    fig = plt.gcf()
    # plt.show()
    # fname = './SSD_data_test/SSD_graphs/SSD_DBSCAN-' + str(ndim) + '-dims.png'
    # fig.savefig(fname)
    plt.clf()
    return clustering, fig

def generate_AgglomerativeCluster(vector, labels, twoD_vector, n_cluster =2 , linkage = 'ward' ):
    clustering = AgglomerativeClustering(n_clusters= n_cluster, linkage= linkage).fit(vector)

    # Get DBSCAN cluster labels
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
            alpha =0.4,
        )


    plt.title('Agglomerative with dim ( ' + str(ndims) + ')\nEstimated number of clusters: ' + str(n_clusters_)
              + '\n #clusters = ' + str(n_cluster) + ', linkage = ' + linkage)
    fig = plt.gcf()
    # plt.show()
    fname = './SSD_data_test/SSD_graphs/SSD_agglomerative-' + str(ndims) + '-dims.png'
    # fig.savefig(fname)
    plt.clf()
    return clustering, fig

def generate_spectral_clustering(vector, labels, twoD_vector, n_cluster = 2, assign_labels = 'discretize' ):
    clustering = SpectralClustering(n_clusters= n_cluster, assign_labels = assign_labels, random_state=0).fit(vector)

    # Get DBSCAN cluster labels
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

    plt.title('Spectral with dim ( ' + str(ndims) + ')\nEstimated number of clusters: ' + str(n_clusters_)
              + '\n #clusters = ' + str(n_cluster) + ', labels = ' + assign_labels)
    fig = plt.gcf()
    # plt.show()
    # fname = './Dikedataset_graphs/figs/synthetic-graph-comparison-graph2vec-' + str(ndims) + '-dims.png'
    # fname = './SSD_data_test/SSD_graphs/SSD_agglomerative-' + str(ndims) + '-dims.png'
    # fig.savefig(fname)
    plt.clf()

    return clustering, fig

def generate_kmeans_clustering(vector, labels, twoD_vector, n_cluster = 2, random_state = 0):
    clustering = KMeans(n_clusters=n_cluster, random_state = random_state).fit(vector)

    # Get DBSCAN cluster labels
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


    plt.title('Kmeans with dim ( ' + str(ndims) + ')\n #clusters = ' + str(n_cluster) )
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

def plot_clusters(twoD_vector, cluster_Labels, alg_name ='', hyp_para= 0, ndim= 2):
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

    # for each in labels.keys():
    #     plt.annotate(each,labels[each], weight= 'bold', size = 20)

    plt.title( alg_name + ' with dim ( ' + str(ndim) + ')\n #density para = ' + str(hyp_para/100) )
    fig = plt.gcf()
    plt.show()

    # fname = './SSD_data_test/SSD_graphs/SSD_agglomerative-' + str(ndims) + '-dims.png'
    # fig.savefig(fname)
    plt.clf()
    return fig