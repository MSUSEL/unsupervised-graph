from matplotlib import pyplot as plt
import pandas as pd
import sklearn as sk
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn import inspection
import os
import time
from datetime import timedelta
from matplotlib.offsetbox import AnchoredText
import pickle

def classifier_name_expand(alg):
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
    elif  alg == 'ridge':
        alg = 'Ridge'
    elif alg == 'pa':
        alg = 'Passive Aggressive'
    elif alg == 'sgd':
        alg = 'SGD'
    elif alg == 'perc':
        alg = 'Perceptron'
    #embedding methods
    elif alg == 'g2v':
        alg = 'Graph2Vec'
    elif alg == 'wlksvd':
        alg = 'WL-KSVD'
    elif alg == 'gl2v':
        alg = 'GL2Vec'
    return alg

sem = True

def supervised_methods_evaluation(alg, model, X, y, X_test, y_test, ndim,
                                  X_train_size, y_train_size, output_path, emb, training_time,optimized=False, optimizing_tuple=None):
    
    global sem 
    start_time = time.monotonic()
    acc_scores = cross_val_score(model, X, y, cv = 5, n_jobs = -1 )
    training_time = timedelta(seconds=time.monotonic() - start_time)
    
    start_time = time.monotonic()
    y_pred = model.fit(X,y).predict(X_test)
    prediction_time = timedelta(seconds=time.monotonic() - start_time)
    
    #output_file_path = output_path + "supervised_methods_evaluation_results/" + alg + "/"
    output_path = output_path + "supervised_methods_evaluation_results/" + emb + '/' + alg + "/"

    if optimized:
        output_path = output_path + "/optimized_results/"  
        summary_file_path = output_path + emb + '_' + alg + "_summary_optimized.csv"
        output_file_path = output_path + emb + '_' + alg + "_" + str(ndim) + "-dims_optimized_" + "results.txt"
        roc_data_path = output_path + emb + '_' + alg + "_" + str(ndim) + "-dims_optimized_" + "roc_data.csv"
    else:
        output_path = output_path + "/plain_results/"
        summary_file_path = output_path + emb + '_' + alg + "_summary_plain.csv"
        output_file_path = output_path + emb + '_' + alg + "_" + str(ndim) + "-dims_" + "results.txt"
        roc_data_path = output_path + emb + '_' + alg + "_" + str(ndim) + "-dims_" + "roc_data.csv"

    os.makedirs(os.path.dirname(summary_file_path), exist_ok=True)

    output_file = open(output_file_path, "w")
    output_file2 = open(summary_file_path, "a")
    #output_file3 = open(roc_data_path, "w")
    
    output_file.write("Classifier Name: " + classifier_name_expand(alg))
    output_file2.write(classifier_name_expand(alg) + ", ")

    output_file.write("\nEmbedding Name: " + classifier_name_expand(emb))
    output_file2.write(classifier_name_expand(emb) + ", ")
    
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
                      
    f1Score = sk.metrics.f1_score(y_test, y_pred)
    output_file.write("\nF1 Score: " + str(f1Score))
    output_file2.write(str(f1Score) + ", ")                       

    f2Score = sk.metrics.fbeta_score(y_test, y_pred, beta = 1.2)
    output_file.write("\nF2 Score: " + str(f2Score))
    output_file2.write(str(f2Score) + ", ")                       
    
    while sem == False:
        pass
    print("Dim entering plot writing block: ", ndim)
    sem = False
    cm = sk.metrics.confusion_matrix(y_test, y_pred)
    output_file.write("\nConfusion Matrix: \n" + str(cm))
    cmdisp = sk.metrics.ConfusionMatrixDisplay(cm, display_labels=['Benign', 'Malignant'])
    cmdisp.plot()
    plt.title(classifier_name_expand(alg) + " " + str(ndim) + "-Dimensions Confusion Matrix")
    plt.savefig(output_path + emb + "_" + alg + "_" + str(ndim) + "-dims_" + "cm.png")
    plt.close()
    #plt.show(block=False)
    
    if alg not in ['lsvc', 'ridge', 'pa', 'sgd', 'perc']:
        y_score = model.predict_proba(X_test)
        y_score_rav = y_score[:,1]
    else:
        y_score_rav = model.decision_function(X_test)
    #fig_obj = plt.figure()

    ras = sk.metrics.roc_auc_score(y_test, y_score_rav)
    fpr, tpr, thrsh = sk.metrics.roc_curve(y_test, y_score_rav)
    plt.plot(fpr,tpr,label=str(ndim) + "-Dim")
    plt.title(classifier_name_expand(alg) + " " + str(ndim) + "-Dimensions ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    text_box = AnchoredText("ROC AUC Score: " + str(ras), frameon=True, loc=4, pad=0.5)
    plt.setp(text_box.patch, facecolor='white', alpha=0.5)
    plt.gca().add_artist(text_box)
    plt.savefig(output_path + emb + "_" + alg + "_" + str(ndim) + "-dims_" + "ras.png")
    plt.close()
    
    #this only works with 2-d data
    '''
    dbd = sk.inspection.DecisionBoundaryDisplay.from_estimator(model, X_test, response_method='decision_function')
    dbd.plot()
    plt.title(classifier_name_expand(alg) + " " + str(ndim) + "-Dimensions Decision Boundary")
    plt.savefig(output_path + emb + "_" + alg + "_" + str(ndim) + "-dims_" + "db.png")
    plt.close()
    '''
    
    print("Dim leaving plot writing block: ", ndim)
    sem = True
        
    output_file.write("\nROC AUC Score: " + str(ras))
    output_file2.write(str(ras)+ ", ")
    
    data = pd.DataFrame({ 'fpr' : fpr , 'tpr' : tpr, 'thrsh' : thrsh  })
    data.to_csv(roc_data_path, index = False)
   
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

    return output_path, model

def group_plotter_by_dim(superv_alg_name, emb, ndims_list, output_path, clf_type):
    
    for alg in superv_alg_name:
        plt.title(classifier_name_expand(alg) + " " + str(min(ndims_list)) + "-" + str(max(ndims_list))  + " Dimensions ROC Curves")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        
        if clf_type == 'optimized':
            for dim in ndims_list:
                file_inp = output_path + "supervised_methods_evaluation_results/" + str(emb) + '/' + str(alg) + \
                                            '/optimized_results/' + str(emb) + '_' + str(alg) + '_' + str(dim) + '-dims_optimized_roc_data.csv'  
                fpr = pd.read_csv(file_inp, usecols = ['fpr'])
                tpr = pd.read_csv(file_inp, usecols = ['tpr'])
                thrsh = pd.read_csv(file_inp, usecols = ['thrsh'])
                plt.plot(fpr,tpr,label = str(dim) + "-Dim")
            plt.legend()
            plt.savefig(output_path + "supervised_methods_evaluation_results/" + str(emb) + "/" + str(alg) + \
                                            '/optimized_results/' + str(emb) + "_" + str(alg) + "_roc_data_optimized_dim_summary.png")
            plt.close()
            
        elif clf_type == 'plain':
            for dim in ndims_list:
                file_inp = output_path + "supervised_methods_evaluation_results/" + str(emb) + '/' + str(alg) + \
                                            '/plain_results/' + str(emb) + '_' + str(alg) + '_' + str(dim) + '-dims_roc_data.csv'  
                fpr = pd.read_csv(file_inp, usecols = ['fpr'])
                tpr = pd.read_csv(file_inp, usecols = ['tpr'])
                thrsh = pd.read_csv(file_inp, usecols = ['thrsh'])
                plt.plot(fpr,tpr,label = str(dim) + "-Dim")
            plt.legend()
            plt.savefig(output_path + "supervised_methods_evaluation_results/" + str(emb) + "/" + str(alg) + \
                                            '/plain_results/' + str(emb) + "_" + str(alg) + "_roc_data_plain_dim_summary.png")
            plt.close()
            
    return 0

def group_plotter_by_alg(superv_alg_name, emb, ndims_list, output_path, clf_type):
    print(output_path)
    for dim in ndims_list:
        plt.title( str(dim) + "-Dimensions ROC Curves by Classifier")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        
        if clf_type == 'optimized':
            for alg in superv_alg_name:
                file_inp = output_path + "supervised_methods_evaluation_results/" + str(emb) + '/' + str(alg) + \
                                            '/optimized_results/' + str(emb) + '_' + str(alg) + '_' + str(dim) + '-dims_optimized_roc_data.csv'  
                fpr = pd.read_csv(file_inp, usecols = ['fpr'])
                tpr = pd.read_csv(file_inp, usecols = ['tpr'])
                thrsh = pd.read_csv(file_inp, usecols = ['thrsh'])
                plt.plot(fpr,tpr,label = classifier_name_expand(alg))
            plt.legend()
            plt.savefig(output_path + "supervised_methods_evaluation_results/" + str(emb) + "/" + str(emb) + "_" + str(dim) +
                        "-dim_summary_roc_data_optimized_dim_summary.png")
            plt.close()
            
        elif clf_type == 'plain':
            for alg in superv_alg_name:
                file_inp = output_path + "supervised_methods_evaluation_results/" + str(emb) + '/' + str(alg) + \
                                            '/plain_results/' + str(emb) + '_' + str(alg) + '_' + str(dim) + '-dims_roc_data.csv'  
                fpr = pd.read_csv(file_inp, usecols = ['fpr'])
                tpr = pd.read_csv(file_inp, usecols = ['tpr'])
                thrsh = pd.read_csv(file_inp, usecols = ['thrsh'])
                plt.plot(fpr,tpr,label = classifier_name_expand(alg))
            plt.legend()
            plt.savefig(output_path + "supervised_methods_evaluation_results/" + str(emb) + "/" + str(emb) + "_" + str(dim) + "-dim_roc_data_plain_dim_summary.png")
            plt.close()
            
    return 0