# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 08:13:04 2021

@author: Albert
"""

import numpy as np
import umap
from sklearn import decomposition
from sklearn import cluster
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
import keras
import random
from itertools import chain
from joblib import Parallel, delayed
import wittgenstein as w
import time
import pandas
import os
from foil import foil, foil_cutRules, foil_predict, foil_evaluate, multi_prediction_prob



# =============================================================================
#   General Preprocessing (equal for all labels; i.e. has to be done only once)
# Input: Original data, i.e. training examples, test examples 
#       Method used for feature extraction (UMAP, PCA or pretrained NN)
#       Option to apply normalization as well as thresholding
# Output: Training and test data set in the form which is necessary for our approach, 
#         i.e. (binary) vectors ;
#        Optional: Feature extraction yielding appropriate data for clustering
# =============================================================================

def preprocess(train_data, test_data, normalization = True, fe_method = None, UMAPsetting = None, model_name = None, thresholding = True, threshold = 0.1):
     #vectorization
    total_train = train_data.shape[0]
    total_test = test_data.shape[0]
    
    length = 1
    for i in range(1, len(train_data.shape)):
        length *= train_data.shape[i]
    
    train = np.reshape(train_data, (total_train, length)) 
    test = np.reshape(test_data, (total_test, length))
    
    
    #normalization
    if (normalization):
        train = train / np.max(train)
        test = test / np.max(test)
    
    
    #optional application of a Feature-Extraction-Method
    if (fe_method == 'UMAP'):
        if (UMAPsetting == None):
            n_neighbors = 5
            min_dist = 0.05
            n_components = 3
            random_state = 42
        else:
            n_neighbors, min_dist, n_components, random_state = UMAPsetting
            
        clustering_data = umap.UMAP(n_neighbors = n_neighbors,
                                    min_dist = min_dist,
                                    n_components = n_components,
                                    random_state = random_state,
        ).fit_transform(train)
    elif(fe_method == 'PCA'):
        pca = decomposition.PCA(n_components = 50)
        pca.fit(train)
        clustering_data = pca.transform(train)
    elif(fe_method == 'NN'):
        model = keras.models.load_model(model_name)
        shape = (train.shape[0],) + model.input_shape[1:]
        data = train.reshape(shape)
        
        intermediate_layer_model = keras.Model(inputs=model.input,
                                       outputs=model.get_layer(index = -2).output)
        clustering_data = intermediate_layer_model.predict(data)
        
    else:
        clustering_data = train
        
        
    #thresholding
    if (thresholding):
        train = (train > threshold) * 1
        test = (test > threshold) * 1
    
    return train, test, clustering_data


# =============================================================================
#   Clustering
# Input: Clustering_data, train data, label to be predicted +
#        chosen method (DBSCAN, k-means, ...) as well as corresponding parameters 
#       + optionality for clustering neg. examples
# Output: Clusters of positive examples +  negative representatives
# =============================================================================
  
def clustering(clustering_data, train, train_label, pred_label, cl_method = 'k_means', num_pos_clusters = 3, num_neg_clusters = 1000, neg_clustering = False, cl_method_neg = 'x_means', apply_clustering = True):
    #separate positive and negative data
    pos = train[train_label == pred_label,:]
    neg = train[train_label != pred_label,:]
    cl_pos = clustering_data[train_label == pred_label,:]
    cl_neg = clustering_data[train_label != pred_label,:]
    
    clusters_pos = []
    
    if (not apply_clustering):
        clusters_pos.append(pos)
        return clusters_pos, neg
    
    if (cl_method == 'k_means'):
        kmeans = cluster.KMeans(n_clusters = num_pos_clusters).fit(cl_pos)
        labels_pos = kmeans.labels_
        
        for i in range(num_pos_clusters):
            clusters_pos.append(pos[labels_pos == i,:])
    
    elif(cl_method == 'x_means'):
        amount_initial_centers = 2
        initial_centers_pos = kmeans_plusplus_initializer(cl_pos, amount_initial_centers).initialize()
     
        xmeans_instance = xmeans(cl_pos, initial_centers_pos, kmax = num_pos_clusters).process()
        labels_pos = xmeans_instance.get_clusters()
        
        for i in range(len(labels_pos)):
            clusters_pos.append(pos[labels_pos[i],:])
        
    elif (cl_method == 'DBSCAN'):
        dbscan_instance = cluster.DBSCAN(eps=0.25, min_samples=30, n_jobs = -1).fit(cl_pos)
        labels_pos = dbscan_instance.labels_
        
        num_pos_cls = len(np.unique(labels_pos)) - 1
        if (num_pos_cls < 2):
            print("Clustering failed! Positive examples have not been changend.")
            num_pos_cls = 1
            clusters_pos.append(pos)
        else:
            print("{} clusters generated.".format(num_pos_cls))
            for i in range(np.minimum(num_pos_cls,8)):
                clusters_pos.append(pos[labels_pos == i,:])
            
 
    
    #optional: apply clustering also on negative examples:
    if (neg_clustering):
        if (cl_method_neg == 'k_means'):
            kmeans_neg = cluster.KMeans(n_clusters = num_neg_clusters).fit(cl_neg)
            labels_neg = kmeans_neg.labels_
            
            clusters_neg = []
            for i in range(num_neg_clusters):
                clusters_neg.append(neg[labels_neg == i,:])
        
        elif (cl_method_neg == 'x_means'):
            amount_initial_centers = 2
            initial_centers_neg = kmeans_plusplus_initializer(cl_neg, amount_initial_centers).initialize()
         
            xmeans_instance_neg = xmeans(cl_neg, initial_centers_neg, kmax = num_neg_clusters).process()
            labels_neg = xmeans_instance_neg.get_clusters()
            
            clusters_neg = []
            for i in range(len(labels_neg)):
                clusters_neg.append(neg[labels_neg[i],:])    
            num_neg_clusters = len(clusters_neg)
            
        elif (cl_method_neg == 'DBSCAN'):
            dbscan_instance_neg = cluster.DBSCAN(eps=0.35, min_samples=100, n_jobs = -1).fit(cl_neg)
            labels_neg = dbscan_instance_neg.labels_
            
            clusters_neg = []
            num_neg_cls = len(np.unique(labels_neg)) - 1
            if (num_neg_cls < 2):
                print("Clustering failed! Negative examples have not been changend.")
                num_neg_clusters = 1
                clusters_neg.append(neg)
            else:
                print("{} clusters generated.".format(num_neg_cls))
                for i in range(num_neg_cls):
                    clusters_neg.append(neg[labels_neg == i,:])
                num_neg_clusters = len(clusters_neg)
        
        num_representants = int(neg.shape[0] / (4 * num_neg_clusters))
    
        neg_representatives = []
        for i in range(num_neg_clusters):
            rows = random.sample(range(clusters_neg[i].shape[0]), np.min((num_representants,len(clusters_neg[i]))))
            neg_representatives.append(clusters_neg[i][rows,:])
    
        negs = np.array(list(chain.from_iterable(neg_representatives)))
        
    else:
        negs = neg
        
    return clusters_pos, negs



# =============================================================================
#   Foil-Application
# Input: clusters_pos, negs, maximal number of learned rules
# Output: Rules + print(results) 
# =============================================================================
    
#FIRST WE DEFINE A HELP FUNCTION
#this function should check whether a new rule is already part of
#a given basis set of rules or a specification of a 'basis' rule
#and return an update of the basis set by adding new rules or exchanging 
#existing rules by their specifications
def comp_rules(basis, new):
    new_org = np.reshape(new, (new.shape[0], new.shape[1], 1))
    n1 = np.sum(new[:, 0] != -10) - 1 #relevant partial rules
    new = new[0:n1,:]
    idx1 = new[:,0]
    for i in range(basis.shape[2]):
        redundant = True
        rule = np.copy(basis[:,:,i])
        n2 = np.sum(rule[:, 0] != -10) - 1
        rule = rule[0:n2,:]
        idx2 = rule[:,0]
        if n2 < n1: #change role of new and rule
            temp = np.copy(new)
            new = np.copy(rule)
            rule = np.copy(temp)
            temp = np.copy(idx1)
            idx1 = np.copy(idx2)
            idx2 = np.copy(temp)
            temp = n1
            n1 = n2
            n2 = n1
        if (not np.all([idx1[k] in idx2 for k in range(n1)])):
            continue
        #comparison of partial rules
        for j in range(n1):
            a = new[j,:]
            if (not np.any(np.all(a == rule[0:n2,:], axis = 1))):
                redundant = False
                break
        if redundant == True:
            #print('This rule is redundant.')
            basis[0:n2,:,i] = rule
            return basis
    
    return np.concatenate((basis, new_org), axis = 2)


#generating the rules corresponding to a cluster (for parallelization)
def myClusterFun_foil(idx, clusters_pos, negs, partial_rules, max_rules, pred_col):
    foil_data = np.concatenate((clusters_pos[idx], negs))
    target = np.concatenate((np.ones(clusters_pos[idx].shape[0]), np.zeros(negs.shape[0])))
    nr = np.reshape(np.array([int(i) for i in np.linspace(0, foil_data.shape[0] - 1, foil_data.shape[0])]), (foil_data.shape[0], 1))
    foil_data = np.c_[foil_data, target, nr]
    
    
    rules = foil(foil_data, partial_rules, max_rules, pred_col)
    rules = foil_cutRules(rules, max_rules)
    if (len(rules.shape) != 3):
        rules = np.reshape(rules, ((partial_rules + 1), 2, 1))
    return rules
    

def foilApplication(clusters_pos, negs, max_rules = 20, combine_rules = False, apply_clustering = True):
    
    pred_col = negs.shape[1]
    partial_rules = np.minimum(20,pred_col)
    
    if (not (apply_clustering)):
        foil_data = np.concatenate((clusters_pos[0], negs))
        target = np.concatenate((np.ones(clusters_pos[0].shape[0]), np.zeros(negs.shape[0])))
        nr = np.reshape(np.array([int(i) for i in np.linspace(0, foil_data.shape[0] - 1, foil_data.shape[0])]), (foil_data.shape[0], 1))
        foil_data = np.c_[foil_data, target, nr]
        basis = foil(foil_data, partial_rules, max_rules, pred_col)
        basis = foil_cutRules(basis, max_rules)
        
        print("Number of learned rules: ", basis.shape[2])
              
        return basis
    
    
    R = Parallel(n_jobs=len(clusters_pos))(delayed(myClusterFun_foil)(i, clusters_pos, negs, partial_rules, max_rules, pred_col) for i in range(len(clusters_pos)))
        
    
        
    if combine_rules:
        basis = R[0]
        rest = np.concatenate(R[1::], axis = 2)
        
        for e in range(rest.shape[2]):
            basis = comp_rules(basis, rest[:,:,e])
    
    else:
        basis = np.concatenate(R[::], axis = 2)
              
    return basis

# =============================================================================
#   Application of RIPPER/IREP
# Input: algorithm (IREP or RIPPER), train + label, clustering data, label to be predicted
# Output: list of classifiers
# =============================================================================

#for parallelization: define function for fitting the classifier on a positive cluster
def myClusterFun_ripper(index, clusters_pos, negs, algorithm):
    data = np.concatenate((clusters_pos[index], negs))
    target = np.concatenate((np.ones(clusters_pos[index].shape[0]), np.zeros(negs.shape[0])))
    if (algorithm == "RIPPER"):
        clf = w.RIPPER()
    elif (algorithm == "IREP"):
        clf = w.IREP()
    clf.fit(data, target, pos_class = 1)
    
    return clf

def fit_clf(clusters_pos, negs, algorithm = 'RIPPER', apply_clustering = True):
    if apply_clustering:
        clf_list = Parallel(n_jobs=len(clusters_pos))(delayed(myClusterFun_ripper)(i, clusters_pos, negs, algorithm) for i in range(len(clusters_pos)))
    else:
        clf_list = []
        data = np.concatenate((clusters_pos[0], negs))
        target = np.concatenate((np.ones(clusters_pos[0].shape[0]), np.zeros(negs.shape[0])))
        if (algorithm == "RIPPER"):
            clf = w.RIPPER()
        elif (algorithm == "IREP"):
            clf = w.IREP()
        clf.fit(data, target, pos_class = 1)
        clf_list.append(clf)
        
        #summaries list of classifiers to one final classifier for the given label (implicite by data)
    classifier = clf_list[0]
    for i in range(len(clf_list) - 1):
        for j in range(len(clf_list[i + 1].ruleset_)):
            classifier.add_rule(clf_list[i+1].ruleset_[j])
        
    
    return classifier

# =============================================================================
#   Binary Classifier
# Input: Data from basic preprocessing + label + algorithm (FOIL, RIPPER, IREP)
# Output: Learned rules or fitted classifiers, respectively
# =============================================================================
        
def binClassifier(algorithm, train, train_label, pred_label, clustering_data, cl_method = 'k_means',
                  num_pos_clusters = 3, num_neg_clusters = 1000, neg_clustering = False, cl_method_neg = 'x_means', apply_clustering = True,
                  max_rules = 20, combine_rules = False):
    
    clusters_pos, negs = clustering(clustering_data, train, train_label, pred_label, cl_method, num_pos_clusters, num_neg_clusters, neg_clustering, cl_method_neg, apply_clustering)    
    
    if (len(clusters_pos) == 1):
        apply_clustering = False
    
    if (algorithm == 'FOIL'):
        return foilApplication(clusters_pos, negs, max_rules, combine_rules, apply_clustering)

    elif (algorithm  == 'RIPPER') | (algorithm == 'IREP'):
        return fit_clf(clusters_pos, negs, algorithm, apply_clustering)
        

# =============================================================================
#   Multiclass Classifier
# Input: Data from basic preprocessing + algorithm(FOIL, RIPPER, IREP)
# Output: List of learned binary classifiers for each label + time consumption
# =============================================================================

def MCClassifier(algorithm, train, train_label, clustering_data, cl_method = 'k_means',  
                 num_pos_clusters = 3, num_neg_clusters = 1000, neg_clustering = False, cl_method_neg = 'x_means',
                 max_rules = 20, combine_rules = False, apply_clustering = True):
    
    labels = np.unique(train_label)
    mcc = []
    times = []
    counter = 0 #in case 'labels' contains something different than ints, we use this variable as loop counter
    startAll = time.time()
    for pred_label in labels:
        start = time.time()
        mcc.append(binClassifier(algorithm, train, train_label, pred_label, clustering_data, cl_method,
                                     num_pos_clusters, num_neg_clusters, neg_clustering, cl_method_neg, apply_clustering, max_rules,
                                     combine_rules))
        end = time.time()
        times.append(end- start)
        counter += 1
        print("\nFinished label ", counter, "/", len(labels), ". ")
        
    endAll = time.time()
    times.append(endAll - startAll)
    print("Total time consumption: ", endAll - startAll)


    return mcc, times


# =============================================================================
#   Prediction:
# Input: test data, learned classifier/rules + algorithm
# Output: predicted label + number of unique predictions
# =============================================================================


def predict_foil(test, mcRules):
    
    result = np.zeros((test.shape[0], len(mcRules)))
    for i in range(len(mcRules)):
        result[:,i] = foil_predict(test, mcRules[i])
    
    prediction = []
    unique = 0
   
    #calculate prediction
    for i in range(test.shape[0]):
        if (np.sum(result[i,:]) == 1):
            unique += 1
            prediction.append(np.argmax(result[i,:]))
            
        elif (np.sum(result[i,:]) == 0): #no binary classifier returns True
            temp = []
            for j in range(len(mcRules)):
                temp.append(multi_prediction_prob(test[i,:], mcRules[j]))
            if (np.sum(np.equal(temp, np.max(temp))) > 1):
                m = np.max(temp)
                for k in range(len(mcRules)):
                    if temp[k] == m:
                        temp[k] = multi_prediction_prob(test[i,:], mcRules[k], add_rules = True)
                if (np.sum(np.equal(temp, np.max(temp))) > 1):
                    prediction.append(-1)
                else:
                    unique += 1
                    prediction.append(np.argmax(temp))
            else:
                 unique += 1
                 prediction.append(np.argmax(temp))
        else: #more than one binary classifier returns True or 1, respectively
            temp = []
            for j in range(len(mcRules)):
                if (result[i,j] == True):
                    temp.append(multi_prediction_prob(test[i,:], mcRules[j], add_rules = True))
                else:
                    temp.append(0)
            if (np.sum(np.equal(temp, np.max(temp))) > 1):
                prediction.append(-1)
            else:
                 unique += 1
                 prediction.append(np.argmax(temp))
                 
    return prediction, unique
   

def predbin_proba(clf, input_vector, add_length = False):
    
    probs, reasons = clf.predict_proba(np.reshape(input_vector, (1,len(input_vector))), give_reasons = True)
    pred = probs[0,1]
    
    if (add_length & (pred > 0.9)):
        pred += np.max([len(reasons[0][i]) for i in range(len(reasons[0]))])
    
    return pred


def predict_ripper(mcclf, test):
    
    predArray = np.zeros((test.shape[0], len(mcclf)))
    for i in range(len(mcclf)):
        predArray[:,i] = mcclf[i].predict(test)
        
    unique = 0   
    unique_pred = []
    for i in range(test.shape[0]):
        if (np.sum(predArray[i,:]) == 1): #exactly one binary classifier returns True
            unique += 1
            unique_pred.append(np.argmax(predArray[i,:])) 
        elif (np.sum(predArray[i,:]) == 0): #no binary classifier returns True
            temp = []
            for j in range(len(mcclf)):
                temp.append(predbin_proba(mcclf[j], test[i,:]))
            if (np.sum(np.equal(temp, np.max(temp))) > 1):
                m = np.max(temp)
                for k in range(len(mcclf)):
                    if temp[k] == m:
                        temp[k] = predbin_proba(mcclf[k], test[i,:], add_length = True)
                if (np.sum(np.equal(temp, np.max(temp))) > 1):
                    unique_pred.append(-1)
                else:
                    unique += 1
                    unique_pred.append(np.argmax(temp))
            else:
                 unique += 1
                 unique_pred.append(np.argmax(temp))
        else: #more than one binary classifier returns True
            temp = []
            for j in range(len(mcclf)):
                if (predArray[i,j] == 1):
                    temp.append(predbin_proba(mcclf[j], test[i,:], add_length = True))
                else:
                    temp.append(0)
            if (np.sum(np.equal(temp, np.max(temp))) > 1):
                unique_pred.append(-1)
            else:
                 unique += 1
                 unique_pred.append(np.argmax(temp))  
                 
    return unique_pred, unique


def predict(algorithm, mcc, test):
    if (algorithm == 'FOIL'):
        return predict_foil(test, mcc)

    elif (algorithm  == 'RIPPER') | (algorithm == 'IREP'):
        return predict_ripper(mcc, test)
 

# =============================================================================
#   Evaluation of a learned multi-class classifier
# Input: Classifier + applied algorithm, test data
# Output: various metrics of accuracy 
# =============================================================================

def table(prediction, truth):
    l = []
    for i in range(len(prediction)):
        l.append([prediction[i],truth[i]])
    
    df = pandas.DataFrame(l, columns = ('prediction', 'truth'))
    tab = df.groupby(['prediction','truth']).size().unstack()
    
    if (-1 in prediction): 
        return tab.drop(labels = -1)
    else:
        return tab

def evaluation_foil(test, test_label, mcRules, prediction = None, unique = None):
    accs = []
    precs = []
    recs = []
    label = np.unique(test_label)
    divisor = test_label.shape[0]
    for i in range(len(mcRules)):
        rules = mcRules[i]
        acc, prec_t, prec_f, rec_t, rec_f = foil_evaluate(test, rules, test_label == label[i])
        #print ('Final results for label ', i, ': acc ', acc, ' prec ', prec_t, '/', prec_f, ' rec ', rec_t, '/', rec_f)
        accs.append(acc)
        precs.append((np.sum(test_label != label[i]) / divisor) * prec_t + (np.sum(test_label == label[i]) / divisor) * prec_f)
        recs.append((np.sum(test_label != label[i]) / divisor) * rec_t + (np.sum(test_label == label[i]) / divisor) * rec_f)
      
    
    if (prediction == None):
        prediction, unique = predict_foil(test, mcRules)
        
    tab = table(prediction, test_label)
    diag = np.sum(np.diag(tab))
    accs.append(diag / divisor)
    accs.append(diag / unique)
    precs.append(np.sum(np.diag(tab) / np.sum(tab,1)) / len(label))
    recs.append(np.sum(np.diag(tab) / np.sum(tab,0)) / len(label))
    
    return accs, precs, recs


def evaluation_ripper(test, test_label, mcc, prediction = None, unique = None):
    accs = []
    precs = []
    recs = []
    divisor = test_label.shape[0]
    label = np.unique(test_label)
    
    for i in range(len(mcc)):
        clf = mcc[i]
        pred = clf.predict(test)
        truth = test_label == label[i]
        tab = table(pred, truth)
        accs.append(np.sum(np.diag(tab)) / divisor)
        prec_temp = np.diag(tab) / np.sum(tab,1)
        precs.append((np.sum(truth) / divisor) * prec_temp[0] + (np.sum(~truth) / divisor) * prec_temp[1])
        rec_temp = np.diag(tab) / np.sum(tab,0)
        recs.append((np.sum(truth) / divisor) * rec_temp[0] + (np.sum(~truth) / divisor) * rec_temp[1])
        
    if (prediction == None):
        prediction, unique = predict_ripper(mcc, test)
        
    tab = table(prediction, test_label)
    diag = np.sum(np.diag(tab))
    accs.append(diag / divisor)
    accs.append(diag / unique)
    precs.append(np.sum(np.diag(tab) / np.sum(tab,1)) / len(label))
    recs.append(np.sum(np.diag(tab) / np.sum(tab,0)) / len(label))
    
    return accs, precs, recs


def evaluation(algorithm, test, test_label, mcc, prediction = None, unique = None):
    if (algorithm == 'FOIL'):
        return evaluation_foil(test, test_label, mcc, prediction, unique)

    elif (algorithm  == 'RIPPER') | (algorithm == 'IREP'):
        return evaluation_ripper(test, test_label, mcc, prediction, unique)
  
# =============================================================================
# Save rules
# =============================================================================
def save_rules(algorithm, mcc, file):
    if algorithm == 'FOIL':
        if not (os.path.isdir(file)):
            os.makedirs(file)
        for i in range(len(mcc)):
            temp = mcc[i].reshape((mcc[i].shape[0],2*mcc[i].shape[2]), order = 'f')
            temp[temp == -10] = None
            for j in range(0,temp.shape[1],2):
                for k in range(temp.shape[0]):
                    if np.isnan(temp[k,j]):
                        temp[k-1,j] = None
                        temp[k-1:,j+1] = None
            temp = pandas.DataFrame(temp)
            temp.to_excel(file + "/label_" + str(i) + ".xlsx")
        
        
        
    elif (algorithm  == 'RIPPER') | (algorithm == 'IREP'):
        if not (os.path.isdir(file)):
            os.makedirs(file)
        for i in range(len(mcc)):
            temp = np.array(mcc[i].ruleset_)
            open_file = open(file + "/label_" + str(i) + ".txt", "w")
            for j in range(temp.shape[0]):
                open_file.write(str(temp[j]) + '\n')
            open_file.close()

# =============================================================================
# Combine all steps to one algorithm
# =============================================================================

def clrbl(algorithm, train_data, test_data, train_label, test_label, normalization = True, fe_method = None, 
          UMAPsetting = None, model_name = None, thresholding = True, threshold = 0.1, cl_method = 'k_means',  
          num_pos_clusters = 3, num_neg_clusters = 100, neg_clustering = False, cl_method_neg = 'x_means',
          max_rules = 20, combine_rules = False, apply_clustering = True, save_result = None):
    
    train, test, clustering_data = preprocess(train_data, test_data, normalization, fe_method, UMAPsetting, model_name, thresholding, threshold)
    
    mcc, times = MCClassifier(algorithm, train, train_label, clustering_data, cl_method, num_pos_clusters, num_neg_clusters,
                              neg_clustering, cl_method_neg, max_rules, combine_rules, apply_clustering)
    
    if save_result != None:
        save_rules(algorithm, mcc, save_result)
    
    accs, precs, recs = evaluation(algorithm, test, test_label, mcc)
    
    accs = list(map(lambda x: x *100, accs))
    precs = list(map(lambda x: x *100, precs))
    recs = list(map(lambda x: x *100, recs))
    
    return np.round(times,1), np.round(accs,2), np.round(precs,2), np.round(recs,2)



