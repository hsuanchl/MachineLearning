import csv
import sys
import numpy as np

import logging
logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)

def train(train_input,index,train_out,metrics_out):
    num_state1_res1 = 0 				#of democrats that answered yes
    num_state1_res2 = 0 				#of republicans that answered yes
    num_state2_res1 = 0 				#of democrats that answered no
    num_state2_res2 = 0 				#of republicans that answered no
    correct = 0 					    #of correct predictions
    wrong = 0
    state1 = ""
    state2 = ""
    res1 = ""
    res2 = ""

    with open(train_input) as tsvfile:
        #load tsv and save to numpy array
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        title = np.array(next(tsvreader))   			#first row is title
        dataWithLabel = np.array([l for l in tsvreader])
        label = dataWithLabel[:,-1]     			    #last row is label
        data = dataWithLabel[:,:-1]     			    #remove label from data
        # print("title",title.shape)
        # print('data',data.shape)
        # print('label',label.shape)

        #find attribute states and result(lables)
        state1 = data[0,0]
        res1 = label[0]
        for x in np.nditer(data):   				#loop through 2D array data
            if x != state1:
                state2 = x
                break
        for x in label:             				#loop through 1D array label
            if x != res1:
                res2 = x
                break
        logging.debug(' s1: %s' %state1)
        logging.debug(' s2: %s' %state2)
        logging.debug(' r1: %s' %res1)
        logging.debug(' r2: %s' %res2)

        for line in dataWithLabel:
            # print (line)
            # split data into yes and no, then perform majority vote
            if line[index] == state1:
                if line[-1]==res1:
                    num_state1_res1 += 1
                elif line[-1]==res2:
                    num_state1_res2 += 1
            if line[index] == state2:
                if line[-1]==res1:
                    num_state2_res1 += 1
                elif line[-1]==res2:
                    num_state2_res2 += 1
        # set the majority to be answer for yes and no
        if num_state1_res1 > num_state1_res2:
            pred_state1 = res1
        else:
            pred_state1 = res2
        if num_state2_res1 > num_state2_res2:
            pred_state2 = res1
        else:
            pred_state2 = res2

        #output train label file
        file = open(train_out,"w")
        for line in dataWithLabel:
            predict = ""
            if line[index] == state1:
                predict = pred_state1
            elif line[index] == state2:
                predict = pred_state2
            if predict != "":
                file.write(predict)
                file.write("\n")
            #calculate error
            if predict == line[-1]:
                correct += 1
            else:
                wrong += 1
        file.close()
        #output error to metric file
        error = wrong/(correct+wrong)
        file = open(metrics_out,"w")
        file.write("error(train): ")
        file.write(str(error))
        file.close()
        print("error(train)",error)
        return pred_state1,pred_state2,state1,state2

def test(pred_state1,pred_state2,state1,state2,test_input,index,test_out,metrics_out):
    correct = 0
    wrong = 0
    with open(test_input) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        title = np.array(next(tsvreader))   			#first row is title
        dataWithLabel = np.array([l for l in tsvreader])
        
        #output test label file
        file = open(test_out,"w")
        for line in dataWithLabel:
            predict = ""
            if line[index] == state1:
                predict = pred_state1
            elif line[index] == state2:
                predict = pred_state2
            if predict != "":
                file.write(predict)
                file.write("\n")
            #calculate error
            if predict == line[-1]:
                correct += 1
            else:
                wrong += 1
        file.close()
        #output error to metric file
        error = wrong/(correct+wrong)
        file = open(metrics_out,"a")  				#a is append, w is write
        file.write("\n")
        file.write("error(test): ")
        file.write(str(error))
        file.close()
        print("error(test)",error)

if __name__ == '__main__':
    #arguments
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    split_index = int(sys.argv[3])  				#convert string to int
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics_out = sys.argv[6]

    #train and test
    pred_state1, pred_state2, state1, state2 = train(train_input,split_index,train_out,metrics_out)
    test(pred_state1,pred_state2,state1,state2,test_input,split_index,test_out,metrics_out)
