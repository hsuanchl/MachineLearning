import numpy as np
import sys
from feature import load_dict
import matplotlib.pyplot as plt

#to prevent overflow, use the equation that makes the exp term positive
def sigmoid(x):
    if x < 0 :
        return 1/(1+np.exp(-x))
    else:
        return np.exp(x)/(np.exp(x)+1)

#takes in list theta and dictionary xi and perform sparse dot product
def sparse_dot(theta,xi):
    product = 0.0
    for i,v in xi.items():
        product += theta[i] * v
    return product

#SGD for only one example. returns theta (Mx1 vector)
def SGD_one_example(theta,lr,xi,yi):       #xi is Mx1, yi is scalar
    xi_vector = np.zeros(len(theta))       #convert dictionary into vector form
    for k,v in xi.items():
        xi_vector[k] = v
    theta = theta + lr * xi_vector * (yi - sigmoid(sparse_dot(theta.T,xi)))
    return theta

#returns labels y and features x (x is a list of dictionaries, x[i] is the sparse dictionary representation for the ith example)
def load_data(path):     
    file = open(path, "r") 
    x = []
    y = []
    for line in file:
        line = line.rstrip('\n')
        y.append(int(line[0]))
        features = dict(x.split(":") for x in line[2:].split('\t')) 
        features = dict([int(k), int(v)] for k, v in features.items())  #convert string to int
        x.append(features)
    file.close()
    return x,y

#since bias is folded in theta, we need to add 1 into last term of x
def fold_bias(x,M):
    for dic in x:
        dic[M] = 1
    return x

#predict label for one example xi after training theta
def predict(theta,xi):
        if sparse_dot(theta.T,xi) > 0:
            return 1
        else:
            return 0

#returns a list of predicted labels and the loss
def prediction_error(theta,x,y):
    pred_list = []
    wrong_count = 0
    for i in range(len(y)):
        prediction = predict(theta,x[i])      
        pred_list.append(prediction)
        if prediction != y[i]:
            wrong_count +=1
    loss = wrong_count/len(y)
    return pred_list, loss

#writes predicted labels to file
def output_label(output_path, pred_list):
    file = open(output_path,"w")
    for y in pred_list:
        file.write(str(y))
        file.write("\n")
    file.close()

#writes error to file
def output_error(output_path,error_train,error_test):
    file = open(output_path,"w")
    file.write("error(train): ")
    file.write("{0:.6f}".format(error_train))   #display 6 decimal palces
    file.write("\n")
    file.write("error(test): ")
    file.write("{0:.6f}".format(error_test))
    file.write("\n")
    file.close()

def neg_log_likelihood(x,y,theta):
    J = 0
    for xi,yi in zip(x,y):
        J += -yi * (sparse_dot(theta.T,xi)) + np.log(1 + np.exp(sparse_dot(theta.T,xi)))
    return J

if __name__== "__main__":
    #Arguments
    formatted_train_input = sys.argv[1]
    formatted_validation_input = sys.argv[2]
    formatted_test_input = sys.argv[3]
    dict_input = sys.argv[4]
    train_out = sys.argv[5]
    test_out = sys.argv[6]
    metrics_out = sys.argv[7]
    num_epoch = int(sys.argv[8])

    #Training
    dictionary = load_dict(dict_input)
    M = len(dictionary)                             # number of features
    train_x,train_y = load_data(formatted_train_input)    
    train_x = fold_bias(train_x,M)
    valid_x,valid_y = load_data(formatted_validation_input)    
    valid_x = fold_bias(valid_x,M)
    theta = np.zeros(M+1)                           #initialize all params with 0

    #SGD
    lr = 0.1
    Javg_list_train = []
    Javg_list_valid = []
    for epoch in range(num_epoch):
        for i in range(len(train_y)):
            theta = SGD_one_example(theta,lr,train_x[i],train_y[i])
        J_avg_train = neg_log_likelihood(train_x,train_y,theta) /len(train_y)
        J_avg_valid = neg_log_likelihood(valid_x,valid_y,theta) /len(valid_y)
        Javg_list_train.append(J_avg_train)
        Javg_list_valid.append(J_avg_valid)
    print(theta)


    #Testing
    test_x,test_y = load_data(formatted_test_input)
    test_x = fold_bias(test_x,M)

    pred_list_train, error_train = prediction_error(theta,train_x,train_y)
    pred_list_test, error_test = prediction_error(theta,test_x,test_y)

    output_label(train_out, pred_list_train)
    output_label(test_out, pred_list_test)
    output_error(metrics_out,error_train,error_test)

    print("train error",error_train)
    print("test error",error_test)


    plt.plot(range(num_epoch),Javg_list_train,label="Training")
    plt.plot(range(num_epoch),Javg_list_valid,label="Validation")
    plt.legend()
    plt.title("Average Negative Log Likelihood - Model 1")
    plt.xlabel("Epoch")
    plt.ylabel("Average Negative Log Likelihood")
    plt.show()