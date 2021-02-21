import csv
import sys
import numpy as np

import logging
logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.WARNING)
# logger.setLevel(logging.ERROR)
logger.setLevel(logging.CRITICAL)

class Node:
    def __init__(self,data):
        self.data = data
        self.left = None
        self.right = None
        self.splitIndexList = []
        self.splitIndexCurrent = None
        self.attributeState = None
        self.depth = 0
        self.pred_label = None

def load_tsv(train_input):
    #load tsv and save to numpy array
    with open(train_input) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        title = np.array(next(tsvreader))                   #first row is title
        dataWithLabel = np.array([l for l in tsvreader])    #title has been removed
        label = dataWithLabel[:,-1]                         #last row is label
    #loop through 1D array label to find the second label
    lab1 = label[0]
    lab2 = ""
    for x in label:             
        if x != lab1:
            lab2 = x
            break
    return dataWithLabel, lab1, lab2, title

def count_Labels(data,lab1,lab2):
    count1=np.count_nonzero(data[:,-1] == lab1)
    count2=np.count_nonzero(data[:,-1] == lab2)
    return count1, count2
    
def gini_impurity(data):
    n = data.shape[0]
    if len(data)==0:
        return 0
    count1=np.count_nonzero(data[:,-1] == data[0,-1])
    count2=np.count_nonzero(data[:,-1] != data[0,-1])
    gini_imp = count1/n * (1-count1/n) + count2 /n * (1-count2/n)
    logging.debug('\n %s' %data)
    logging.debug('number of examples: %s' %n)
    logging.debug('count1: %s' %count1)
    logging.debug('count2: %s' %count2)
    logging.debug('gini impururity: %s' %gini_imp)
    return gini_imp

def gini_gain(node,index):
    gini_imp = gini_impurity(node.data)
    data = node.data[:,[index,-1]]                                      #data is now only columns with the selected attribute and label
    logging.debug('data with just the selected attribute:\n%s' %data)
    n = data.shape[0]
    count1=np.count_nonzero(data[:,0] == data[0,0])
    count2=np.count_nonzero(data[:,0] != data[0,0])
    # logging.debug('count1: %s' %count1)
    # logging.debug('count2: %s' %count2)
    dataState1 = data[data[:,0]==data[0,0]]
    dataState2 = data[data[:,0]!=data[0,0]]
    # logging.debug('data attribute with just State1:\n%s' % dataState1)
    # logging.debug('data attribute with just State2:\n%s' % dataState2)
    gini_imp1 = gini_impurity(dataState1)
    gini_imp2 = gini_impurity(dataState2)
    giniGain = gini_imp - (count1/n * gini_imp1 + count2/n * gini_imp2)
    return giniGain

def train_stump(node):
    logging.warning('calling stump')
    if len(node.splitIndexList) == node.data.shape[1]-1 or gini_impurity(node.data)==0 or node.depth >= max_depth:
        return
    data = node.data
    m = data.shape[1]-1                                                 #number of attributes (need to minus one because last column is label)
    logging.warning('number of total attributes:%s' %m)
    max_gain = -1
    max_index = -1
    #calculate gini gain of each attribute to determine which to split with
    for i in range(m):
        if i in node.splitIndexList:                                    #if the index is already used to split before, skip it
            continue
        giniGain = gini_gain(node,i)
        logging.error('gini gain for attribute index %s, %s is: %s', i, title[i],giniGain)
        if giniGain > max_gain:
            max_gain = giniGain
            max_index = i
        elif max_index != -1 and giniGain == max_gain:
            if title[i] > title[max_index]:
                max_gain = giniGain
                max_index = i
    if max_gain <= 0:
        return
    logging.warning('max gain, split on index: %s' %max_index)

    #split data and create left and right nodes
    unique, counts = np.unique(data[:,max_index],return_counts=True)
    # dataState1 = data[data[:,max_index]==data[0,max_index]]
    # dataState2 = data[data[:,max_index]!=data[0,max_index]]
    dataState1 = data[data[:,max_index]==unique[0]]
    dataState2 = data[data[:,max_index]==unique[1]]
    node.left = Node(dataState1)
    node.right = Node(dataState2)
    node.left.attributeState = unique[0]
    node.right.attributeState = unique[1]
    node.left.splitIndexList = node.splitIndexList + [max_index]
    node.right.splitIndexList = node.splitIndexList + [max_index]
    node.left.splitIndexCurrent = max_index
    node.right.splitIndexCurrent = max_index
    node.left.depth = node.depth + 1
    node.right.depth = node.depth + 1
    logging.warning('\nnode.left.data:\n%s' %node.left.data)
    logging.warning('\nnode.right.data:\n%s' %node.right.data)
    logging.warning('node.left attributes used: %s' %node.left.splitIndexList)
    logging.warning('node.left attributes used: %s' %node.right.splitIndexList)
    logging.warning('left.depth: %s' %node.left.depth)
    logging.warning('right.depth: %s' %node.right.depth)

    train_stump(node.left)
    train_stump(node.right)
    

def train_tree(train_input):
    data, lab1, lab2, title = load_tsv(train_input)
    logging.warning('\noriginal data:\n%s' %data)
    #create root node, build tree, then label tree
    root = Node(data)
    train_stump(root)
    label_trained_tree(root,lab1,lab2)
    return root

#use majority vote to predict label
def maj_vote(node,lab1,lab2):
    count1, count2 =  count_Labels(node.data,lab1,lab2)
    if count1 > count2:
        pred_lab = lab1
    elif count1 < count2:
        pred_lab = lab2
    else:  #count1 == count2, choose the label that comes last in the lexicographical order
        pred_lab=lab1 if lab1>lab2 else lab2
    return pred_lab

def label_trained_tree(root,lab1,lab2):
    if root.left == None and root.right == None:
       root.pred_label = maj_vote(root,lab1,lab2)
       return
    label_trained_tree(root.left,lab1,lab2)
    label_trained_tree(root.right,lab1,lab2)


#using trained tree, predict of one example (on line of data)
def predict_example(root,line):
    if (root.pred_label != None):
        return root.pred_label
    index = root.left.splitIndexCurrent
    # print('split on: ',index)
    # print('state is: ', line[index])
    if (root.left.attributeState == line[index]):
        # print("travelled left")
        prediction = predict_example(root.left,line)
    else:
        # print("travelled right")
        prediction = predict_example(root.right,line)
    return prediction

#using trained tree, predict data and output to file
def predict_data(root,input_file,output_file):
    data, lab1, lab2, title = load_tsv(input_file)
    file = open(output_file,"w")
    for line in data:
        predict = predict_example(root,line)
        file.write(predict)
        file.write("\n")
    file.close()

def error(train_input,train_out,test_input,test_out,lab11,lab2,metrics_out):
    #load the 4 files
    with open(train_input) as trainInput:
        tsvreader = csv.reader(trainInput, delimiter="\t")
        np.array(next(tsvreader))                           #first row is title
        dataWithLabel = np.array([l for l in tsvreader])    #title has been removed
    train_label = dataWithLabel[:,-1]                       #last row is label
    with open(test_input) as testInput:
        tsvreader = csv.reader(testInput, delimiter="\t")
        np.array(next(tsvreader))                           #first row is title
        dataWithLabel = np.array([l for l in tsvreader])    #title has been removed
        test_label = dataWithLabel[:,-1]                    #last row is label
    train_pred = np.genfromtxt(train_out,dtype='str')
    test_pred = np.genfromtxt(test_out,dtype='str')
    #calculate error and output to file
    error_train = np.count_nonzero(train_label != train_pred) / len(train_label)
    error_test = np.count_nonzero(test_label != test_pred) / len(test_label)
    file = open(metrics_out,"w")
    file.write("error(train): ")
    file.write(str(error_train))
    file.write("\n")
    file.write("error(test): ")
    file.write(str(error_test))
    file.close()
    print("train error:",error_train)
    print("test errror", error_test)
    return error_train, error_test

def printTree(root,title,lab1,lab2):
    if root:
        if root.depth == 0:
            count1,count2 = count_Labels(root.data,lab1,lab2)
            print('['+str(count1)+' '+str(lab1)+'/'+str(count2)+' '+str(lab2)+']')
        else:
            for i in range(root.depth):
                print("|", end =" ")
            print(title[root.splitIndexCurrent], end =" ")
            print("=", root.attributeState, end =": ")
            count1,count2 = count_Labels(root.data,lab1,lab2)
            print('['+str(count1)+' '+str(lab1)+'/'+str(count2)+' '+str(lab2)+']')
            # print('label is:',root.pred_label)
            # print('split on index:',root.splitIndexCurrent)
            # print('\n')

        printTree(root.right,title,label1,label2)
        printTree(root.left,title,label1,label2)

if __name__ == '__main__':
    #arguments
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    max_depth = int(sys.argv[3])
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics_out = sys.argv[6]

    data, label1, label2,title = load_tsv(train_input)
    root = train_tree(train_input)
    printTree(root,title,label1, label2)
    #output files for train data prediciton, test data prediction, and error
    predict_data(root,train_input,train_out)
    predict_data(root,test_input,test_out)
    error(train_input,train_out,test_input,test_out,label1,label2,metrics_out)



    #question 1.3.2
    if False:
        import matplotlib.pyplot as plt
        train_input = "politicians_train.tsv"
        # train_input = "education_train.tsv"
        test_input = "politicians_test.tsv"
        # test_input = "education_test.tsv"
        max_depth = 1
        train_out = "q1.3train.txt"
        test_out = "q1.3test.txt"
        metrics_out = "q1.3metrics.txt"

        data, label1, label2,title = load_tsv(train_input)
        m = data.shape[1]-1     # number of attributes, -1 because last column is label
        errorTrainList = []
        errorTestList = []
        for i in range(m):
            max_depth = i
            root = train_tree(train_input)
            printTree(root,title,label1, label2)
            #output files for train data prediciton, test data prediction, and error
            predict_data(root,train_input,train_out)
            predict_data(root,test_input,test_out)
            error_train, error_test = error(train_input,train_out,test_input,test_out,label1,label2,metrics_out)
            errorTrainList.append(error_train)
            errorTestList.append(error_test)
        print("train:",errorTrainList)
        print("test:",errorTestList)
        plt.plot(errorTrainList, label='Train Error')
        plt.plot(errorTestList, label='Test Error')
        plt.legend()
        plt.title("Politicians Dataset Train/Test Error")
        plt.xlabel("Max-Depth")
        plt.xlabel("Error")
        # plt.show()

    #individual function tests
    # data, label1, label2,title = load_tsv(train_input)
    # print(data)
    # root = Node(data)
    # gini_gain(root,0)
    # train_stump(root)  
    # label_trained_tree(root,label1,label2)
    # test_tree(trained_tree,train_input)
    # test_tree(trained_tree,test_input)
    # line = data[15,:]
    # print(predict_example(root,line))
