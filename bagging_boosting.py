# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Homework 3 for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.

import numpy as np
import os
import math
from numpy import array
import graphviz
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import random



def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """

    # INSERT YOUR CODE HERE
    d=dict()
    for i in range(0,len(x),1):
        if x[i] in d:
            d[x[i]].append(i)
        else:
            d.setdefault(x[i], [])
            d[x[i]].append(i)
    return d
    raise Exception('Function not yet implemented!')


def entropy(y,w=None):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """

    #INSERT YOUR CODE HERE
    entropy=0
    count_0=0
    count_1=0
    dict1=partition(y)
    for i in range(0,len(y),1):
        if y[i]==list(dict1.keys())[0]:
            count_0+=w[i][0]
        else:
            count_1+=w[i][0]
    p1=count_0/(count_0+count_1)
    p2=count_1/(count_0+count_1)
    entropy=p1*math.log(p1, 2)+p2*math.log(p2,2)
    return -1*entropy
    

def mutual_information(x, y,w=None):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """
    dict1=partition(y)
    entropy_before_split=entropy(y,w)
    mutual_information=entropy_before_split
    x_value_list=np.unique(x)
    entropy_after_split=0
    for x_val in x_value_list:
        count1=0
        count2=0
        countx=0
        for i in range(0,len(x),1):
            if x[i]==x_val and y[i]==list(dict1.keys())[0]:
                count1+=w[i][0]
                countx+=w[i][0]
            elif x[i]==x_val and y[i]==list(dict1.keys())[1]:
                count2+=w[i][0]
                countx+=w[i][0]
        if count1>0:
            p1=(count1)/(count1+count2)
        else:
            p1=1
        if count2>0:
            p2=(count2)/(count1+count2)
        else:
            p2=1
        entropy_after_split+=(countx/np.sum(w))*((-1)*p1*math.log(p1,2)-p2*math.log(p2,2))
    mutual_information-=entropy_after_split
    return mutual_information


def generate_attribute_value_pairs(x):
    attribute_val_list=[]
    for i in range(0,len(x[0]),1):
        for j in x[:,i]:
            dumlist=(i,j)
            if dumlist not in attribute_val_list:
                attribute_val_list.append(dumlist)
    return attribute_val_list


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5,w=None):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    #termination 1
    y_len=np.unique(y)
    if len(y_len)==1:
        #print("condition 1")
        return y_len[0]

    if attribute_value_pairs is None:
        attribute_value_pairs=generate_attribute_value_pairs(x)
    #termination 2 & 3
    if depth==max_depth or len(attribute_value_pairs)==0:
        #print("condition 2")
        count_0=0
        count_1=0
        dict1=partition(y)
        for i in range(0,len(y),1):
            if y[i]==list(dict1.keys())[0]:
                count_0+=w[i]
            else:
                count_1+=w[i]
        if count_0>count_1:
            return list(dict1.keys())[0]
        else:
            return list(dict1.keys())[1]

    
    mutual_info=[]
    for key,value in attribute_value_pairs:
        x_subset=np.array(x[ :,int(key)]==value).astype('int')
        mutual_information_val=mutual_information(x_subset,y,w=w)
        mutual_info.append(mutual_information_val)
    split_on=max(mutual_info)
    split_cred=mutual_info.index(split_on)
    splitting_feature=attribute_value_pairs[split_cred][0]
    splitting_value=attribute_value_pairs[split_cred][1]
    attribute_value_pairs = np.delete(attribute_value_pairs, split_cred, 0)
    data_split=np.array(x[ :,int(splitting_feature)]==int(splitting_value)).astype('int')
    partitioned_dict=partition(data_split)
    if len(partitioned_dict)==2:
        list_of_indexes_0=partitioned_dict[0]
        list_of_indexes_1=partitioned_dict[1]
    elif len(partitioned_dict)==1:
        if list(partitioned_dict.keys())[0]==0:
            list_of_indexes_0=partitioned_dict[0]
            list_of_indexes_1=[]
        else:
            list_of_indexes_1=partitioned_dict[0]
            list_of_indexes_0=[]
    x_false_f = x[list_of_indexes_0, :]
    y_false_f = y[list_of_indexes_0]
    w_false_f=w[list_of_indexes_0]
    x_true_f = x[list_of_indexes_1, :]
    y_true_f = y[list_of_indexes_1]
    w_true_f=w[list_of_indexes_1]

    root={}

    root[(splitting_feature,splitting_value,False)]=id3(x_false_f,y_false_f,attribute_value_pairs=attribute_value_pairs,depth=depth+1,max_depth=max_depth,w=w_false_f)
    root[(splitting_feature,splitting_value,True)]=id3(x_true_f,y_true_f,attribute_value_pairs=attribute_value_pairs,depth=depth+1,max_depth=max_depth,w=w_true_f)

    return root

    raise Exception('Function not yet implemented!')


def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """

    #INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    tree_key=tree.keys()
    for key in tree_key:
        if key[2]==(x[int(key[0])]==int(key[1])):
            if type(tree[key])!=dict:
                return tree[key]
            else: 
                return predict_example(x,tree[key])


def predict_example_boosting(x,h_ens):
    y_predicted=[]
    for item in h_ens:
        tree=item[0]
        y_predicted.append(predict_example(x,tree))
    dict1=partition(y_predicted)
    predicted_label_0=0
    predicted_label_1=1
    for i in range(0,len(y_predicted),1):
        if y_predicted[i]==list(dict1.keys())[0]:
            predicted_label_0+=h_ens[i][1]
        else:
            predicted_label_1+=h_ens[i][1]
    if len(dict1)==2:
        if predicted_label_0>predicted_label_1:
            return list(dict1.keys())[0]
        else:
            return list(dict1.keys())[1]
    else:
        return list(dict1.keys())[0]




            
def compute_error(y_true, y_pred,w=None):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """

    # INSERT YOUR CODE HERE
    sum_val=0
    for i in range(0,len(y_true),1):
        if(y_true[i]!=y_pred[i]):
            sum_val+=1
    return sum_val/len(y_true)
    raise Exception('Function not yet implemented!')


def pretty_print(tree, depth=0):
    """
    Pretty prints the decision tree to the console. Use print(tree) to print the raw nested dictionary representation
    DO NOT MODIFY THIS FUNCTION!
    """
    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1} {2}]'.format(split_criterion[0], split_criterion[1], split_criterion[2]))

        # Print the children
        if type(sub_trees) is dict:
            pretty_print(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


def render_dot_file(dot_string, save_file, image_format='png'):
    """
    Uses GraphViz to render a dot file. The dot file can be generated using
        * sklearn.tree.export_graphviz()' for decision trees produced by scikit-learn
        * to_graphviz() (function is in this file) for decision trees produced by  your code.
    DO NOT MODIFY THIS FUNCTION!
    """
    if type(dot_string).__name__ != 'str':
        raise TypeError('visualize() requires a string representation of a decision tree.\nUse tree.export_graphviz()'
                        'for decision trees produced by scikit-learn and to_graphviz() for decision trees produced by'
                        'your code.\n')

    # Set path to your GraphViz executable here
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    graph = graphviz.Source(dot_string)
    graph.format = image_format
    graph.render(save_file, view=True)


def to_graphviz(tree, dot_string='', uid=-1, depth=0):
    """
    Converts a tree to DOT format for use with visualize/GraphViz
    DO NOT MODIFY THIS FUNCTION!
    """

    uid += 1       # Running index of node ids across recursion
    node_id = uid  # Node id of this node

    if depth == 0:
        dot_string += 'digraph TREE {\n'

    for split_criterion in tree:
        sub_trees = tree[split_criterion]
        attribute_index = split_criterion[0]
        attribute_value = split_criterion[1]
        split_decision = split_criterion[2]

        if not split_decision:
            # Alphabetically, False comes first
            dot_string += '    node{0} [label="x{1} = {2}?"];\n'.format(node_id, attribute_index, attribute_value)

        if type(sub_trees) is dict:
            if not split_decision:
                dot_string, right_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, right_child)
            else:
                dot_string, left_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, left_child)

        else:
            uid += 1
            dot_string += '    node{0} [label="y = {1}"];\n'.format(uid, sub_trees)
            if not split_decision:
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, uid)
            else:
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, uid)

    if depth == 0:
        dot_string += '}\n'
        return dot_string
    else:
        return dot_string, node_id, uid

def plot_graph(Xtrn, ytrn, Xtst, ytst):
    trnErr = dict()
    tst_err= dict()
    for i in range(1,11):
        decision_tree= id3(Xtrn, ytrn, max_depth=i)
        y_pred_trn=[]
        y_pred_tst=[]
        for x in Xtrn:
            y_pred_trn.append(predict_example(x, decision_tree))
        trnErr[i] = compute_error(ytrn, y_pred_trn)
        for x in Xtst:
            y_pred_tst.append(predict_example(x, decision_tree))
        tst_err[i] = compute_error(ytst, y_pred_tst)
    plt.figure()
    plt.plot(trnErr.keys(), trnErr.values(), marker='o', linewidth=3, markersize=12)
    plt.plot(tst_err.keys(), tst_err.values(), marker='s', linewidth=3, markersize=12)
    plt.xlabel('Depth', fontsize=16)
    plt.ylabel('Training/Test error', fontsize=16)
    plt.xticks(list(trnErr.keys()), fontsize=12)
    plt.legend(['Training Error', 'Test Error'], fontsize=16)
    plt.axis([0,11, 0,0.4])
    plt.show()


def bagging(x, y, max_depth, num_trees):
    list_of_samples_x=[]
    list_of_samples_y=[]
    for i in range(0,num_trees,1):
        idx = np.random.choice(np.arange(len(x)),len(x), replace=True)
        x1=[]
        for item in idx:
            x1.append(x[item])
        x1 = np.array(x1)
        list_of_samples_x.append(x1)
        y1=[]
        for item in idx:
            y1.append(y[item])
        y1 = np.array(y1)
        list_of_samples_y.append(y1) 
    d_tree=[]
    for j in range(0,len(list_of_samples_x),1):
        d_tree.append(id3(list_of_samples_x[j], list_of_samples_y[j], max_depth=max_depth,w=np.full((len(x), 1),1/len(x))))
    return d_tree

def boosting(x, y, max_depth, num_stumps):
    #initialize weights
    initials_weights=np.full((len(x), 1),1/len(x))
    
    final_ensemble=[]
    
    #learn a decision stump
    for i in range(0,num_stumps,1):
        tree=id3(x,y,depth=0,max_depth=max_depth,w=initials_weights)
        y_pred_index_correct=[]
        y_pred_index_wrong=[]
        y_predicted=[]
        for i in range(0,len(x),1):
            if predict_example(x[i],tree)!=y[i]:
                y_pred_index_wrong.append(i)
                y_predicted.append(predict_example(x[i],tree))
            else:
                y_pred_index_correct.append(i)
                y_predicted.append(predict_example(x[i],tree))
        
        #calculate weight of tree
        sum_val=0
        sum_of_weights=sum(initials_weights)
        for item in y_pred_index_wrong:
            sum_val+=initials_weights[item][0]
        
        alpha=0.5*math.log((1-sum_val)/sum_val)
        #print(alpha)
        
        temp_arr=[]
        temp_arr.append(tree)
        temp_arr.append(alpha)
        final_ensemble.append(temp_arr)

        sum_of_weights=sum(initials_weights)
        #calulate new weights of correctly classified and misclassified exapmles
        for item in y_pred_index_wrong:
            new_weight_wrong=initials_weights[item]*math.exp(alpha)/sum_of_weights
            initials_weights[item]=new_weight_wrong
        
        for item in y_pred_index_correct:
            new_weight_correct=initials_weights[item]*math.exp(-1*alpha)/sum_of_weights
            initials_weights[item]=new_weight_correct

    return final_ensemble



if __name__ == '__main__':


    # #Bagging
    M = np.genfromtxt('./mushroom.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)

    ytrn = M[:, 0]
    Xtrn = M[:, 1:]
    # print(Xtrn)

    # Load the test data
    M = np.genfromtxt('./mushroom.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    dict_label=partition(ytst)
    labels=[list(dict_label.keys())[0],list(dict_label.keys())[1]]
    print("Bagging Implementation")
    for depth in [3,5]:
        for num_trees in [5,10]:
            tree_ens=bagging(Xtrn,ytrn,depth,num_trees)
            dict1=partition(ytst)
            Y_pred=[]
            for item in Xtst:
                labelcount1=0
                labelcount2=0
                for tree in tree_ens:
                    if predict_example(item,tree)==list(dict1.keys())[0]:
                        labelcount1+=1
                    else:
                        labelcount2+=1
                if(labelcount1>labelcount2):
                    Y_pred.append(list(dict1.keys())[0])
                else:
                    Y_pred.append(list(dict1.keys())[1])
            tst_err_1 = compute_error(ytst, Y_pred)
            print("Depth=",depth,"Bag_size=",num_trees)
            print('Accuracy = {0:4.2f}%.'.format(100-(tst_err_1 * 100)))
            cm=confusion_matrix(ytst, Y_pred, labels)
            print(cm)

    print("For Bagging Scikit")
    #Scikit Bagging
    from sklearn.ensemble import RandomForestClassifier
    for depth in [3,5]:
        for num_trees in [5,10]:
            clf = RandomForestClassifier(n_estimators=num_trees, max_depth=depth,criterion='entropy', random_state=0)
            clf.fit(Xtrn, ytrn)
            y_test_pred = clf.predict(Xtst);
            err_test = clf.score(Xtst, ytst)
            print("Depth=",depth,"Bag_size=",num_trees)
            print('Accuracy= {0:4.2f}'.format((err_test * 100)))
            cm=confusion_matrix(ytst, y_test_pred, labels)
            print(cm)

    #Boosting:
    M = np.genfromtxt('./mushroom.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)

    ytrn = M[:, 0]
    Xtrn = M[:, 1:]
    # print(Xtrn)

    # Load the test data
    M = np.genfromtxt('./mushroom.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]
    print("Boosting Implementation")
    for depth in [1,2]:
        for num_trees in [5,10]:
            tree_ens=boosting(Xtrn, ytrn, depth, num_trees)
            dict1=partition(ytst)
            Y_pred=[]
            for item in Xtst:
                Y_pred.append(predict_example_boosting(item,tree_ens))
            tst_err_1 = compute_error(ytst, Y_pred)
            print("Depth=",depth,"Number_of_Stumps=",num_trees)
            print('Accuracy Boosting= {0:4.2f}'.format(100-(tst_err_1 * 100)))
            labels=[list(dict1.keys())[0],list(dict1.keys())[1]]
            cm=confusion_matrix(ytst, Y_pred, labels)
            print(cm)


    M = np.genfromtxt('./mushroom.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)

    ytrn = M[:, 0]
    Xtrn = M[:, 1:]
    # print(Xtrn)

    # Load the test data
    M = np.genfromtxt('./mushroom.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    dict_label=partition(ytst)
    labels=[list(dict_label.keys())[0],list(dict_label.keys())[1]]
    from sklearn.ensemble import AdaBoostClassifier
    #Boosting with scikitlearn:
    print("For Boosting Scikit")
    for depth in [1,2]:
        for num_stumps in [5,10]:
            base = DecisionTreeClassifier(max_depth=depth)
            clf = AdaBoostClassifier(base_estimator = base, n_estimators=num_stumps)
            clf.fit(Xtrn, ytrn)
            y_test_pred = clf.predict(Xtst);
            err_test = clf.score(Xtst, ytst)
            print("Depth=",depth,"Number_of_Stumps=",num_stumps)
            print('Accuracy= {0:4.2f}'.format((err_test * 100)))
            cm=confusion_matrix(ytst, y_test_pred, labels)
            print(cm)


    





    
    

























