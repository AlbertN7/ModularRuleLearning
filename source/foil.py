# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 14:38:14 2020

@author: Albert
"""
import pandas
import numpy as np

# =============================================================================
#   Define a gain function
#   A rule has the form (column, value)
# =============================================================================
def rule_holds(rule, data):
    return(np.sum(data[:,rule[0]] == rule[1]))

def gain(rule, Pos, Neg):
    p = rule_holds(rule, Pos)
    n = rule_holds(rule, Neg)
    return(p * (np.log2(p / (p + n)) - np.log2(Pos.shape[0] / (Pos.shape[0] + Neg.shape[0]))))


# =============================================================================
#   Define a function generating the basic form of the matrix A
#   containing the gain-values of each possible rule
#   col1...gain_value, col2...column (rule), col3... value (rule)
# =============================================================================

def fill_cols(P):
    rows = 0
    col2 = []
    col3 = []
    for i in range(P.shape[1] - 1):
        s = set(P[:,i])
        n = len(s)
        rows += n
        col2 += list(np.repeat(i,n))
        col3 += list(s)
        
    return rows, col2, col3

def gen_A(P):
    rows, col2, col3 = fill_cols(P)
    
    A = np.zeros([rows, 3])
    A[:,1] = col2
    A[:,2] = col3
    
    return A
    
# =============================================================================
#   Define a function updating a given matrix A 
# =============================================================================

def calc_gain(rule, P, N, taken_cols):
    rule = [int(rule[0]), rule[1]]
    if (rule[0] in taken_cols) | (not (rule[1] in list(set(P[:,rule[0]])))):
        return -1000
    else:
        return gain(rule, P, N)

def update_A(A, P, N, taken_cols):
    temp = pandas.DataFrame(A[:,1:3])
    A[:,0] = list(temp.apply(calc_gain, axis = 1, P = P, N = N, taken_cols = taken_cols))
    
    return A


# =============================================================================
#   Define the FOIL algorithm
# =============================================================================

def foil(data, partial_rules, total_rules, pred_col, pred_val = True):
    #define basic form of a data frame R containg the chosen rules
    R = np.zeros(((partial_rules + 1), 2, total_rules))
    R[:, 0, :] = -10
    
    #define a set of positive examples P
    P = data[data[:,pred_col] == pred_val, :]
    P = np.delete(P, pred_col, 1)
    
    #check whether any positive examples are given
    if (P.shape[0] == 0):
        R[0,:,0] = [-10, data.shape[0]]
        
    #generate basic form of matrix A
    A = gen_A(P)
    
    #initialize a variable counting the number of 'complete rules'
    counter_total = 0
    
    #iterate until P is empty or a maximal number of rules is reached
    while ((P.shape[0] > 0) & (counter_total < total_rules)):
        #define a set of negative examples
        N = data[data[:, pred_col] != pred_val, :]
        N = np.delete(N, pred_col, 1)
        
        #initialize a variable counting the number of 'partial rules'
        counter_partial = 0
        
        #initialize a list storing the columns which are already taken
        taken_cols = []
        
        #define a copy of P used in the following while-loop
        P_help = P
        
        #check whether any negative examples are given 
        #(extrem case: only positive examples as input)
        if (N.shape[0] == 0):
            #return "arbitrary" rule
            rule = [0, P[0,0]]
            R[counter_partial, :, counter_total] = rule
            P_help = P_help[P_help[:, rule[0]] == rule[1], :]
            counter_partial += 1
            
        #iterate until N is empty or a maximal number of partial rules is reached
        while ((N.shape[0] > 0) & (counter_partial < partial_rules)):
            #update A with new gain-values
            A = update_A(A, P_help, N, taken_cols)
            
            #determine 'best rule'
            maxgain = np.argmax(A[:,0])
            col = int(A[maxgain, 1])
            val = A[maxgain, 2]
            rule = [col, val]
            
            #remove examples which do not satisfy the chosen rule
            N = N[N[:, rule[0]] == rule[1], :]
            P_help = P_help[P_help[:, rule[0]] == rule[1], :]
            
            #store the rule in R and update the help variables
            R[counter_partial, :, counter_total] = rule
            taken_cols.append(rule[0])
            counter_partial += 1
            
        #remove elements of P which satisfy the 'complete rule' generated above
        idx = [int(i) for i in P_help[:, -1]]
        t = [P[i,-1] in idx for i in range(P.shape[0])]
        t = [not i for i in t]
        P = P[t]
        
        #store the number of remaining elements of P and N 
        if (P.shape[0] > 0):
            R[counter_partial, :, counter_total] = [P.shape[0], N.shape[0]]
        else:
            R[counter_partial, :, counter_total] = [-1, N.shape[0]]
            
        #update the counter of 'complete rules'
        counter_total += 1
        
    return R
    
# =============================================================================
#   Define functions evaluating the generated rules
# =============================================================================

#define a function which evaluates a single rule on a given input
def single_prediction(input_vector, rule):
    n = np.sum(rule[:, 0] != -10) - 1
    cols = [int(i) for i in rule[0:n, 0]]
    vals = [float(i) for i in rule[0:n, 1]]
    inp = [i for i in input_vector[cols]]
    
    return inp == vals
    
#define a function which evaluates a set of rules
def multi_prediction(input_vector, rules):
    b = []
    for i in range(rules.shape[2]):
        b.append(single_prediction(input_vector, rules[:,:,i]))
    
    return(any(b))
 
#define a function evaluating the results of the FOIL-algorithm on a test set
def foil_predict(data, rules):
    prediction = []
    for i in range(data.shape[0]):
        prediction.append(multi_prediction(data[i,:], rules))
    
    return prediction

#create table of predicted results
def foil_table(data, rules, truth):
    pred = pandas.DataFrame(foil_predict(data, rules))
    pred['NR'] = np.linspace(1,pred.shape[0], pred.shape[0])
    truth = pandas.DataFrame(truth)
    truth['NR'] = np.linspace(1, truth.shape[0], truth.shape[0])
    
    pred_t = pred[pred[0] == True]['NR']
    pred_f = pred[pred[0] == False]['NR']
    
    truth_t = list(truth[truth[0] == True]['NR'])
    truth_f = list(truth[truth[0] == False]['NR'])
    
    tt = np.sum(pred_t.apply(lambda x: x in truth_t)) #pred = T, truth = T
    tf = np.sum(pred_t.apply(lambda x: x in truth_f)) #pred = T, truth = F
    ft = np.sum(pred_f.apply(lambda x: x in truth_t)) #pred = F, truth = T
    ff = np.sum(pred_f.apply(lambda x: x in truth_f)) #pred = F, truth = F
    
    table = np.array([[tt, tf], [ft, ff]])
    
    return table

#more precise results for precision and recall
def foil_evaluate(data, rules, truth):
    table = foil_table(data, rules, truth)
    accuracy = np.sum(np.diag(table)) / np.sum(table)
    
    precision = np.diag(table) / np.sum(table,1)
    recall = np.diag(table) / np.sum(table,0)
    
    precision_t = precision[0]
    precision_f = precision[1]
    recall_t = recall[0]
    recall_f = recall[1]
    
    if np.isnan(np.min(precision)):
        precision_t = 0
        precision_f = 0
    
    if np.isnan(np.min(recall)):
        recall_t = 0
        recall_f = 0
    
    #print(table)
    
    return(accuracy, precision_t, precision_f, recall_t, recall_f)


# =============================================================================
#   Define additional functions modifying the learned rules
# =============================================================================

#remove needless rules
def foil_cutRules(rules, total_rules):
    i = 1
    while (rules[0,0, (total_rules - i)] == -10):
        i += 1
    i -= 1
    
    return rules[:, :, 0:(total_rules - i)]




# =============================================================================
#   Especially for MC-Classifiers it might be helpful to consider probabilities
#   of fulfilling a rule instead of True/False predictions
# =============================================================================

#define a function which evaluates a single rule on a given input
def single_prediction_prob(input_vector, rule):
    n = np.sum(rule[:, 0] != -10) - 1
    cols = [int(i) for i in rule[0:n, 0]]
    vals = [float(i) for i in rule[0:n, 1]]
    inp = [i for i in input_vector[cols]]
    
    return np.equal(inp, vals)
    
#define a function which evaluates a set of rules
def multi_prediction_prob(input_vector, rules, add_rules = False):
    b = []
    n_partial_rules = []
    for i in range(rules.shape[2]):
        p = single_prediction_prob(input_vector, rules[:,:,i])
        n_partial_rules.append(len(p))
        b.append((np.sum(p) / len(p))) 
    
    if add_rules:
        b = b + (b == np.max(b)) * n_partial_rules
    
    return(np.max(b))
 
    
#define a function evaluating the results of the FOIL-algorithm on a test set
def foil_predict_prob(data, rules, add_rules = False):
    prediction = []
    for i in range(data.shape[0]):
        prediction.append(multi_prediction_prob(data[i,:], rules, add_rules))
      
    return prediction