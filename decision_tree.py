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


def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """

    #INSERT YOUR CODE HERE
    entropy=0
    count_0=0
    count_1=0
    dict1=partition(y)
    for val in y:
        if val==list(dict1.keys())[0]:
            count_0+=1
        else:
            count_1+=1
    p1=count_0/(count_0+count_1)
    p2=count_1/(count_0+count_1)
    entropy=p1*math.log(p1, 2)+p2*math.log(p2,2)
    return -1*entropy
    

def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """
    dict1=partition(y)
    entropy_before_split=entropy(y)
    mutual_information=entropy_before_split
    x_value_list=np.unique(x)
    entropy_after_split=0
    for x_val in x_value_list:
        count1=0
        count2=0
        countx=0
        for i in range(0,len(x),1):
            if x[i]==x_val and y[i]==list(dict1.keys())[0]:
                count1+=1
                countx+=1
            elif x[i]==x_val and y[i]==list(dict1.keys())[1]:
                count2+=1
                countx+=1
        if count1>0:
            p1=(count1)/(count1+count2)
        else:
            p1=1
        if count2>0:
            p2=(count2)/(count1+count2)
        else:
            p2=1
        entropy_after_split+=(countx/len(x))*((-1)*p1*math.log(p1,2)-p2*math.log(p2,2))
    mutual_information-=entropy_after_split
    return mutual_information


def generate_attribute_value_pairs(x):
    attribute_val_list=[]
    for i in range(0,len(x[0]),1):
        for j in x[ :,i]:
            dumlist=(i,j)
            if dumlist not in attribute_val_list:
                attribute_val_list.append(dumlist)
    return attribute_val_list


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
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
        return y_len[0]

    if attribute_value_pairs is None:
        attribute_value_pairs=generate_attribute_value_pairs(x)
    #termination 2 & 3
    if depth==max_depth or len(attribute_value_pairs)==0:
        count_0=0
        count_1=0
        dict1=partition(y)
        for item in y:
            if item==list(dict1.keys())[0]:
                count_0+=1
            else:
                count_1+=1
        if count_0>count_1:
            return list(dict1.keys())[0]
        else:
            return list(dict1.keys())[1]

    
    mutual_info=[]
    for key,value in attribute_value_pairs:
        x_subset=np.array(x[ :,int(key)]==value).astype('int')
        mutual_information_val=mutual_information(x_subset,y)
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
    x_true_f = x[list_of_indexes_1, :]
    y_true_f = y[list_of_indexes_1]

    root={}

    root[(splitting_feature,splitting_value,False)]=id3(x_false_f,y_false_f,attribute_value_pairs=attribute_value_pairs,depth=depth+1,max_depth=max_depth)
    root[(splitting_feature,splitting_value,True)]=id3(x_true_f,y_true_f,attribute_value_pairs=attribute_value_pairs,depth=depth+1,max_depth=max_depth)

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
            
def compute_error(y_true, y_pred):
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


    

if __name__ == '__main__':
    # Load the training data
    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    # Learn a decision tree of depth 3
    decision_tree = id3(Xtrn, ytrn, max_depth=3)

    # Pretty print it to console
    pretty_print(decision_tree)

    # Visualize the tree and save it as a PNG image
    dot_str = to_graphviz(decision_tree)
    render_dot_file(dot_str, './my_learned_tree')

    # Compute the test error
    y_pred = [predict_example(x, decision_tree) for x in Xtst]
    tst_err = compute_error(ytst, y_pred)

    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))

    M = np.genfromtxt('monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn1 = M[:, 0]
    Xtrn1 = M[:, 1:]

    # Load the test data 1
    M = np.genfromtxt('monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst1 = M[:, 0]
    Xtst1 = M[:, 1:]


    # Load the training data 2
    M = np.genfromtxt('monks-2.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn2 = M[:, 0]
    Xtrn2 = M[:, 1:]

    # Load the test data 2
    M = np.genfromtxt('monks-2.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst2 = M[:, 0]
    Xtst2 = M[:, 1:]


    # Load the training data 3
    M = np.genfromtxt('monks-3.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn3 = M[:, 0]
    Xtrn3 = M[:, 1:]

    # Load the test data 3
    M = np.genfromtxt('monks-3.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst3 = M[:, 0]
    Xtst3 = M[:, 1:]

    plot_graph(Xtrn1,ytrn1,Xtst1,ytst1)
    plot_graph(Xtrn2,ytrn2,Xtst2,ytst2)
    plot_graph(Xtrn3,ytrn3,Xtst3,ytst3)

    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    # Learn a decision tree of depth 3
    decision_tree_depth1 = id3(Xtrn, ytrn, max_depth=1)
    # Pretty print it to console
    pretty_print(decision_tree_depth1)
    # Visualize the tree and save it as a PNG image
    dot_str = to_graphviz(decision_tree_depth1)
    render_dot_file(dot_str, './my_learned_tree_depth_1')
    # Compute the test error
    y_pred = [predict_example(x, decision_tree_depth1) for x in Xtst]
    dict_part=partition(y_pred)
    labels=[list(dict_part.keys())[0],list(dict_part.keys())[1]]
    cm=confusion_matrix(ytst, y_pred, labels)
    print(cm)
    tst_err1 = compute_error(ytst, y_pred)
    print("Accuracy for my tree for depth 1")
    print('Accuracy = {0:4.2f}%.'.format(100-(tst_err1 * 100)))

    decision_tree_depth3 = id3(Xtrn, ytrn, max_depth=3)
    # Pretty print it to console
    pretty_print(decision_tree_depth3)
    # Visualize the tree and save it as a PNG image
    dot_str = to_graphviz(decision_tree_depth3)
    render_dot_file(dot_str, './my_learned_tree_depth_3')
    # Compute the test error
    y_pred_3 = [predict_example(x, decision_tree_depth1) for x in Xtst]
    dict_part_3=partition(y_pred_3)
    labels_3=[list(dict_part_3.keys())[0],list(dict_part_3.keys())[1]]
    cm_3=confusion_matrix(ytst, y_pred_3, labels_3)
    print(cm_3)
    tst_err2 = compute_error(ytst, y_pred_3)
    print("Accuracy for my tree of depth 3")
    print('Accuracy = {0:4.2f}%.'.format(100-(tst_err2 * 100)))

    decision_tree_depth5 = id3(Xtrn, ytrn, max_depth=5)
    # Pretty print it to console
    pretty_print(decision_tree_depth5)
    # Visualize the tree and save it as a PNG image
    dot_str = to_graphviz(decision_tree_depth5)
    render_dot_file(dot_str, './my_learned_tree_depth_5')
    # Compute the test error
    y_pred_5 = [predict_example(x, decision_tree_depth5) for x in Xtst]
    dict_part_5=partition(y_pred_5)
    labels_5=[list(dict_part_5.keys())[0],list(dict_part_5.keys())[1]]
    cm_5=confusion_matrix(ytst, y_pred_5, labels_5)
    print(cm_5)
    tst_err3 = compute_error(ytst, y_pred_5)
    print("Accuracy for my tree of depth 5")
    print('Accuracy = {0:4.2f}%.'.format(100-(tst_err3 * 100)))




    #Sckits Implementation
    decision_tree_depth1_sckit= DecisionTreeClassifier( criterion = "entropy", random_state = 100, max_depth = 1)
    decision_tree_depth1_sckit.fit(Xtrn, ytrn)
    y_pred_sckit_1  = decision_tree_depth1_sckit.predict(Xtst)
    print("Confusion Matrix for sklearn depth1: ", confusion_matrix(ytst, y_pred_sckit_1)) 
    print ("Accuracy : ", accuracy_score(ytst,y_pred_sckit_1)*100)
    

    decision_tree_depth2_sckit= DecisionTreeClassifier( criterion = "entropy", random_state = 100, max_depth = 3)
    decision_tree_depth2_sckit.fit(Xtrn, ytrn)
    y_pred_sckit_2 = decision_tree_depth2_sckit.predict(Xtst)
    print("Confusion Matrix for sklearn depth3: ", confusion_matrix(ytst, y_pred_sckit_2)) 
    print ("Accuracy : ", accuracy_score(ytst,y_pred_sckit_2)*100)
    
 
    decision_tree_depth3_sckit= DecisionTreeClassifier( criterion = "entropy", random_state = 100, max_depth = 5)
    decision_tree_depth3_sckit.fit(Xtrn, ytrn)
    y_pred_sckit_3 = decision_tree_depth3_sckit.predict(Xtst)
    print("Confusion Matrix for sklearn depth5: ", confusion_matrix(ytst, y_pred_sckit_3)) 
    print ("Accuracy : ", accuracy_score(ytst,y_pred_sckit_3)*100)
    



    #My own dataset Breast Cancer

    M = np.genfromtxt('./wdbc_trn.csv', missing_values=0, skip_header=0, delimiter=',')
    ytrn_1 = M[:, 0]
    Xtrn_1 = M[:, 1:]
    avg = np.mean(Xtrn_1, axis = 0)
    rows, cols = np.shape(Xtrn_1)
    for col in range(cols):
        for row in range(rows):
            if Xtrn_1[row, col] > avg[col]:
                Xtrn_1[row, col] = 1
            else:
                Xtrn_1[row, col] = 0
    ytrn_1=ytrn_1.astype(int)
    Xtrn_1=Xtrn_1.astype(int)


    # Load the test data
    M = np.genfromtxt('./wdbc_tst.csv', missing_values=0, skip_header=0, delimiter=',')
    ytst_1 = M[:, 0]
    Xtst_1 = M[:, 1:]
    avg1 = np.mean(Xtst_1, axis = 0)
    rows, cols = np.shape(Xtst_1)
    for col in range(cols):
        for row in range(rows):
            if Xtst_1[row, col] > avg1[col]:
                Xtst_1[row, col] = 1
            else:
                Xtst_1[row, col] = 0
    ytst_1=ytst_1.astype(int)
    Xtst_1=Xtst_1.astype(int)
    

    decision_tree_depth1 = id3(Xtrn_1, ytrn_1, max_depth=1)
    # Pretty print it to console
    pretty_print(decision_tree_depth1)
    # Visualize the tree and save it as a PNG image
    dot_str = to_graphviz(decision_tree_depth1)
    render_dot_file(dot_str, './my_learned_tree_depth1_My_data_set')
    # Compute the test error
    y_pred = [predict_example(x, decision_tree_depth1) for x in Xtst_1]
    dict_part=partition(y_pred)
    labels=[list(dict_part.keys())[0],list(dict_part.keys())[1]]
    cm1=confusion_matrix(ytst_1, y_pred, labels)
    print(cm1)
    tst_err1 = compute_error(ytst_1, y_pred)
    print("Accuracy for my tree for depth 1")
    print('Accuracy = {0:4.2f}%.'.format(100-(tst_err1 * 100)))

    #Sckits Implementation
    decision_tree_depth1_sckit= DecisionTreeClassifier( criterion = "entropy", random_state = 100, max_depth = 1)
    decision_tree_depth1_sckit.fit(Xtrn_1, ytrn_1)
    y_pred_sckit_1  = decision_tree_depth1_sckit.predict(Xtst_1)
    print("Confusion Matrix for sklearn depth1: ", confusion_matrix(ytst_1, y_pred_sckit_1)) 
    print ("Accuracy : ", accuracy_score(ytst_1,y_pred_sckit_1)*100)

    
    decision_tree_depth3_mine = id3(Xtrn_1, ytrn_1, max_depth=3)
    # Pretty print it to console
    pretty_print(decision_tree_depth3_mine)
    # Visualize the tree and save it as a PNG image
    dot_str = to_graphviz(decision_tree_depth3_mine)
    render_dot_file(dot_str, './my_learned_tree_depth3_My_data_set')
    # Compute the test error
    y_pred = [predict_example(x, decision_tree_depth3_mine) for x in Xtst_1]
    dict_part=partition(y_pred)
    labels=[list(dict_part.keys())[0],list(dict_part.keys())[1]]
    cm1=confusion_matrix(ytst_1, y_pred, labels)
    print(cm1)
    tst_err1 = compute_error(ytst_1, y_pred)
    print("Accuracy for my tree for depth 3")
    print('Accuracy = {0:4.2f}%.'.format(100-(tst_err1 * 100)))

    #Sckits Implementation
    decision_tree_depth3_sckit= DecisionTreeClassifier( criterion = "entropy", random_state = 100, max_depth = 3)
    decision_tree_depth3_sckit.fit(Xtrn_1, ytrn_1)
    y_pred_sckit_3  = decision_tree_depth3_sckit.predict(Xtst_1)
    print("Confusion Matrix for sklearn depth3: ", confusion_matrix(ytst_1, y_pred_sckit_3)) 
    print ("Accuracy : ", accuracy_score(ytst_1,y_pred_sckit_3)*100)

    
    decision_tree_depth5_mine = id3(Xtrn_1, ytrn_1, max_depth=5)
    # Pretty print it to console
    pretty_print(decision_tree_depth5_mine)
    # Visualize the tree and save it as a PNG image
    dot_str = to_graphviz(decision_tree_depth5_mine)
    render_dot_file(dot_str, './my_learned_tree_depth5_My_data_set')
    # Compute the test error
    y_pred = [predict_example(x, decision_tree_depth5_mine) for x in Xtst_1]
    dict_part=partition(y_pred)
    labels=[list(dict_part.keys())[0],list(dict_part.keys())[1]]
    cm1=confusion_matrix(ytst_1, y_pred, labels)
    print(cm1)
    tst_err1 = compute_error(ytst_1, y_pred)
    print("Accuracy for my tree for depth 5")
    print('Accuracy = {0:4.2f}%.'.format(100-(tst_err1 * 100)))

    #Sckits Implementation
    decision_tree_depth5_sckit= DecisionTreeClassifier( criterion = "entropy", random_state = 100, max_depth = 5)
    decision_tree_depth5_sckit.fit(Xtrn_1, ytrn_1)
    y_pred_sckit_5  = decision_tree_depth5_sckit.predict(Xtst_1)
    print("Confusion Matrix for sklearn depth3: ", confusion_matrix(ytst_1, y_pred_sckit_5)) 
    print ("Accuracy : ", accuracy_score(ytst_1,y_pred_sckit_5)*100)
























