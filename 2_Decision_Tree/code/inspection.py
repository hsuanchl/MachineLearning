import csv
import sys
import numpy as np
import logging
logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)

def gini_impurity(input,output):
    #load tsv and save to numpy array
    with open(input) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        title = np.array(next(tsvreader))                   #first row is title
        dataWithLabel = np.array([l for l in tsvreader])
        label = dataWithLabel[:,-1]                         #last row is label

    n = label.shape[0]                                      #number of examples
    count1 = 0
    count2 = 0
    lab1 = label[0]
    lab2 = ""
    pred_lab = ""

    for x in label:                                         #loop through 1D array label to find the second label
        if x != lab1:
            lab2 = x
            break

    for x in label:                                         #loop through 1D array label to count how many label1 and label2
        if x == lab1:
            count1 += 1
        elif x == lab2:
            count2 += 1

    logging.debug(label)
    logging.debug('number of examples: %s' %n)
    logging.debug('lable1: %s' %lab1)
    logging.debug('lable2: %s' %lab2)
    logging.debug('count1: %s' %count1)
    logging.debug('count2: %s' %count2)

    # calculate gini impurtiy and error using majority vote
    gini_imp = count1/n * (1-count1/n) + count2/n * (1-count2/n)
    if count1 > count2:
        pred_lab = lab1
    elif count1 < count2:
        pred_lab = lab2
    else:   #count1 == count2
        pred_lab=lab1 if lab1>lab2 else lab2
    error=count2/n if count1 >= count2 else count1/n

    logging.debug('predicted label: %s' %pred_lab)
    logging.debug('gini impururity: %s' %gini_imp)
    logging.debug('error: %s' %error)
    
    #write to file
    file = open(output,"w")  #a is append, w is write
    file.write("gini_impurity: ")
    file.write(str(gini_imp))
    file.write("\n")
    file.write("error: ")
    file.write(str(error))
    file.close()
    print("gini_impurity:", gini_imp)
    print("error:", error)

if __name__ == '__main__':
    input = sys.argv[1]
    output = sys.argv[2]
    gini_impurity(input,output)
