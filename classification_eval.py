from reader import*
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from karateclub.graph_embedding import Graph2Vec, GL2Vec, SF, IGE

from WL_KSVD import WL_KSVD

from pathlib import Path

import pandas as pd
import numpy as np
import os
from joblib import dump, load
import pickle

from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.inspection import DecisionBoundaryDisplay


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

if __name__ == "__main__":
    print('Graph embedding unsupervised clustering')

    # list of Datastet
    # ds_list = ["ENZYMES", "IMDB-BINARY", "IMDB-MULTI", "NCI1", "NCI109", "PTC_FM", "PROTEINS", "REDDIT-BINARY",
    #            "YEAST", "YEASTH", "UACC257", "UACC257H", "OVCAR-8", "OVCAR-8H", "ZINC_full", "alchemy_full", "QM9"]

    # ds_list = ["Yeast", "YeastH", "UACC257", "UACC257H", "OVCAR-8", "OVCAR-8H"]
    ds_list = ["MUTAG",  "NCI1", "NCI109", "PTC_FM", "PTC_MR", "PROTEINS", "ENZYMES"]
    ds_list = [ "IMDB-BINARY", "IMDB-MULTI"]

    # path for the output vector embeddings
    emb_path = "./Embeddings/"
    clf_path = "./Classifier_data/"

    save_clf = True


    G_emb_list = ["G2V", "GL2V", "SF", "WL_KSVD"]
    ndims_list = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

    clf_names = [
        "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
        "Gaussian Process",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "QDA",
    ]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
    ]

    datasets = ["PTC_FM", "PTC_MR", "PROTEINS", "ENZYMES"]
    # datasets = ["MUTAG"]

    # ndims_list = [2, 4]
    # G_emb_list = ["GKSVD"]

    n_rows = len(G_emb_list)
    n_cols =  len(ndims_list)

    SC_val = np.zeros([n_rows, n_cols])
    SC_test = np.zeros([n_rows, n_cols])


    figure = plt.figure(figsize=(27, 9))
    figure.suptitle('Classification decision boundaries on '+ str(datasets[0]) + ' dataset with ' + str(ndims_list[0]) + ' dims')
    i = 1

    KFolds = 5

    score_test_ds = {}
    score_val_ds = {}
    for ds_cnt, ds_name in enumerate(datasets):
        print(ds_name)

        score_test_fold = {}
        score_val_fold = {}
        for k in range(KFolds):
            n_KFold = str(k)
            print(k)

            score_test_dim = {}
            score_val_dim = {}
            for ndim_cnt, n_dims in enumerate(ndims_list):
                print(n_dims)

                score_test_emb = {}
                score_val_emb = {}
                for G_emb_cnt, G_emb_name in enumerate(G_emb_list):
                    print(G_emb_name)




                # fig_cnt = G_emb_cnt
                # print(fig_cnt)



                    y_label = str(n_dims)

                    data_path = emb_path + ds_name + '/Fold_'+ n_KFold + '/' + G_emb_name + '/'

                    vector_path = data_path + 'train' +'/emb_vectors_' + str(n_dims) + '.csv'
                    label_path = data_path + 'train' + '/labels_' + str(n_dims) + '.csv'

                    test_vector_path = data_path + 'test' + '/emb_vectors_' + str(n_dims) + '.csv'
                    test_label_path = data_path + 'test' + '/labels_' + str(n_dims) + '.csv'

                    vector = pd.read_csv(vector_path, header=None).values
                    test_vector = pd.read_csv(test_vector_path, header=None).values


                    X = StandardScaler().fit_transform(vector)
                    X_test = StandardScaler().fit_transform(test_vector)

                    train_df = pd.read_csv(label_path)
                    vector_labels = train_df['Label'].tolist()

                    test_df = pd.read_csv(test_label_path)
                    y_test = test_df['Label'].tolist()

                    X_train, X_val, y_train, y_val = train_test_split(
                        X, vector_labels, test_size=0.2, random_state=0)

                    # if n_dims == 2:
                        # x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
                        # y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
                        #
                        # # just plot the dataset first
                        # cm = plt.cm.RdBu
                        # # cm_bright = ListedColormap(["#FF0000", "#0000FF"])
                        # ax = plt.subplot(n_rows, n_cols, i)
                        # if fig_cnt == 0:
                        #     ax.set_title("Input data")
                        # # Plot the training points
                        # ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, alpha = 0.4, edgecolors="k")
                        # # Plot the testing points
                        # ax.scatter(
                        #     X_val[:, 0], X_val[:, 1], c=y_val, alpha=0.6, edgecolors="k"
                        # )
                        # ax.set_xlim(x_min, x_max)
                        # ax.set_ylim(y_min, y_max)
                        # ax.set_xticks(())
                        # ax.set_yticks(())
                        # ax.set_ylabel(y_label)
                    i += 1

                    score_test_clf = {}
                    score_val_clf = {}
                    c = 0
                    for clf_name, clf in zip(clf_names, classifiers):
                        print(ds_name + ' fold : ' + n_KFold +', dim : ' + str(n_dims) +', Embedding : '+ G_emb_name)
                        print(clf_name)

                        clf.fit(X_train, y_train)

                        val_score = clf.score(X_val, y_val)

                        print (val_score)
                        score_val_clf[clf_name] = val_score
                        # SC_val [G_emb_cnt][ndim_cnt]= val_score
                        c+=1

                        test_score = clf.score(X_test, y_test)

                        score_test_clf[clf_name] = test_score
                        # SC_test[G_emb_cnt][ndim_cnt] = test_score

                        print(test_score)

                        model_path = clf_path + '/' + ds_name +  '/Fold_'+ n_KFold +'/' + G_emb_name + '/dim_' + str(n_dims) +\
                                     '/Models/'
                        os.makedirs(model_path, exist_ok=True)
                        # Its important to use binary mode
                        modelPickle = open(model_path+ 'model_' + clf_name, 'wb')
                        # source, destination
                        pickle.dump(clf, modelPickle)
                        # close the file
                        modelPickle.close()

                        # load the model from disk
                        # loaded_model = pickle.load(open('knnpickle_file', 'rb'))
                        # result = loaded_model.predict(X_test)

                        # if n_dims == 2:
                        #     ax = plt.subplot(n_rows, n_cols, i)
                        #     DecisionBoundaryDisplay.from_estimator(
                        #         clf, X, alpha=0.8, ax=ax, eps=0.5
                        #     )
                        #
                        #
                        #     # Plot the training points
                        #     ax.scatter(
                        #         X_train[:, 0], X_train[:, 1], c=y_train, alpha = 0.4, edgecolors="k"
                        #     )
                        #     # Plot the testing points
                        #     ax.scatter(
                        #         X_val[:, 0],
                        #         X_val[:, 1],
                        #         c=y_val,
                        #         edgecolors="k",
                        #         alpha=0.6,
                        #     )
                        #
                        #     ax.set_xlim(x_min, x_max)
                        #     ax.set_ylim(y_min, y_max)
                        #     ax.set_xticks(())
                        #     ax.set_yticks(())
                        #     if fig_cnt == 0:
                        #         ax.set_title(clf_name)
                        #     ax.text(
                        #         x_max - 0.3,
                        #         y_min + 0.3,
                        #         ("%.2f" % score).lstrip("0"),
                        #         size=15,
                        #         horizontalalignment="right",
                        #     )
                        i += 1
                    score_test_emb[G_emb_name] = score_test_clf
                    score_val_emb[G_emb_name] = score_test_clf

                score_test_dim[n_dims] = score_test_emb
                score_val_dim[n_dims] = score_test_emb

            score_test_fold[n_KFold] = score_test_dim
            score_val_fold[n_KFold] = score_test_dim

        score_path = clf_path + '/' + ds_name + '/Score/'
        os.makedirs(score_path, exist_ok=True)
        score_test_file_folds = open(score_path + "score_test_" + ds_name + ".pkl", "wb")
        pickle.dump(score_val_fold, score_test_file_folds)
        score_test_file_folds.close()

        score_val_file_folds = open(score_path + "score_val_" + ds_name + ".pkl", "wb")
        pickle.dump(score_val_fold, score_val_file_folds)
        score_val_file_folds.close()


        score_test_ds[ds_name] = score_test_fold
        score_val_ds[ds_name] = score_test_fold
    # plt.tight_layout()
    # plt.show()
    # figure.savefig(figure,'Sample.png')
    # plt.clf()
    print ('done')
