import sys
import numpy as np
import matplotlib.pyplot as plt

def load_tsv(path):
    x = []
    y = []
    file = open(path, "r") 
    for line in file:
        line = line.rstrip('\n').rstrip(' ')
        line = line.split(',')
        x.append(line[1:])                  
        y.append(line[0])               #first column is label
    file.close()
    x = np.asarray(x).astype(np.float)
    y = np.asarray(y).astype(np.int)
    y_onehot = np.zeros((y.shape[0],10))
    for n in range(y.shape[0]):
        y_onehot[n,y[n]] = 1
    return x,y,y_onehot

def output_label(output_path, pred_list):
    file = open(output_path,"w")
    for y in pred_list:
        file.write(str(y))
        file.write("\n")
    file.close()

def cross_entropy_loss(NeuralNetwork, N ,x_bias_all,y_onehot_all):
    Jtotal = 0
    for i in range(N):
        x = np.expand_dims(x_bias_all[i],axis=1)        # one example
        y = np.expand_dims(y_onehot_all[i],axis=1) 
        NeuralNetwork.forward(x,y)
        Jtotal += NeuralNetwork.J
    return Jtotal/N

def output_metrics(output_path,train_J_list,test_J_list,train_error,test_error):
    file = open(output_path,"w")
    for i in range(len(train_J_list)):
        file.write("epoch=")
        file.write(str(i+1))
        file.write(" ")
        file.write("crossentropy(train): ")
        file.write(str(train_J_list[i]))
        file.write("\n")
        file.write("epoch=")
        file.write(str(i+1))
        file.write(" ")
        file.write("crossentropy(test): ")
        file.write(str(test_J_list[i]))
        file.write("\n")
    file.write("error(train): ")
    file.write(str(train_error))
    file.write("\n")
    file.write("error(test): ")
    file.write(str(test_error))

    file.close()


class NeuralNetwork:
    def __init__(self,K,M,num_epoch,hidden_units,init_flag,learning_rate):
        self.num_epoch = num_epoch
        self.hidden_units = hidden_units
        self.init_flag = init_flag
        self.learning_rate = learning_rate
        self.K = K      #number of classes
        self.M = M      #number of attributes
        self.alpha = np.zeros((hidden_units, self.M+1))
        self.beta = np.zeros((self.K, hidden_units+1))
        if init_flag == '1':
            self.alpha[:, 1:] = np.random.uniform(-0.1, 0.1, (hidden_units, self.M))
            self.beta[:, 1:] = np.random.uniform(-0.1, 0.1, (self.K, hidden_units))

    def forward(self,x,y):
        self.x = x
        self.y = y
        self.a = self.linearFor(self.x,self.alpha)
        self.z_no_bias = self.sigmoidFor(self.a)
        self.z = np.expand_dims(np.insert(self.z_no_bias, 0, 1),axis=1)
        self.b = self.linearFor(self.z,self.beta)
        self.y_hat = self.softmaxFor(self.b)
        self.J = self.crossEntropyFor(self.y,self.y_hat)                
        
    def linearFor(self,x,w):
        return np.dot(w,x)

    def sigmoidFor(self,a):
        return 1/(1+np.exp(-a))

    def softmaxFor(self,b):
        return np.exp(b) / np.sum(np.exp(b))

    def crossEntropyFor(self,y,y_hat):
        return -np.sum(y*np.log(y_hat))

    def backward(self):
        self.gb = self.softmaxBack(self.y_hat,self.y)
        self.gbeta,self.gz = self.linearBack1(self.z,self.b,self.gb)
        self.ga = self.sigmoidBack(self.a,self.z_no_bias,self.gz)
        self.galpha = self.linearBack2(self.x,self.a,self.ga)

    def softmaxBack(self,y_hat,y):
        gb = y_hat - y
        return gb

    def linearBack1(self,z,b,gb):
        gbeta = np.dot(gb,z.T)
        gz = self.beta[:,1:].T @ gb
        gz = np.dot(self.beta[:,1:].T, gb)
        return gbeta, gz

    def linearBack2(self,x,a,ga):
        galpha = np.dot(ga,x.T)
        return galpha

    def sigmoidBack(self,a,z,gz):
        ga = z*(1-z)*gz
        return ga

    def update_weights(self):
        self.alpha = self.alpha - self.learning_rate * self.galpha
        self.beta = self.beta - self.learning_rate * self.gbeta



if __name__== "__main__":
    #arguments
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    train_output = sys.argv[3]
    test_output = sys.argv[4]
    metrics_out = sys.argv[5]
    num_epoch = int(sys.argv[6])
    hidden_units = int(sys.argv[7])
    init_flag = sys.argv[8]
    learning_rate = float(sys.argv[9])

    #load training data
    train_x_all,train_y_all,train_y_onehot_all = load_tsv(train_input)    #matrix of all examples
    train_x_bias_all = np.hstack((np.expand_dims(np.ones(train_x_all.shape[0]),axis=1),train_x_all))  #add 1 in the first element for bias term
    train_N = train_x_all.shape[0]      # num of examples
    
    #load test data
    test_x_all,test_y_all,test_y_onehot_all = load_tsv(test_input)        #matrix of all examples
    test_x_bias_all = np.hstack((np.expand_dims(np.ones(test_x_all.shape[0]),axis=1),test_x_all))  #add 1 in the first element for bias term
    test_N = test_x_all.shape[0]      # num of examples

    #define model
    K = 10
    M = train_x_all.shape[1]          # num of attributes
    NN = NeuralNetwork(K,M,num_epoch,hidden_units,init_flag,learning_rate)

    #training
    train_J_list = []
    test_J_list = []
    for e in range(num_epoch):
        for i in range(train_N):
            train_x = np.expand_dims(train_x_bias_all[i],axis=1)        # one example
            train_y = np.expand_dims(train_y_onehot_all[i],axis=1)      # one example

            #forward backward pass
            NN.forward(train_x,train_y)
            NN.backward()
            NN.update_weights()

        #calc loss
        train_Javg = cross_entropy_loss(NN,train_N,train_x_bias_all,train_y_onehot_all)
        test_Javg = cross_entropy_loss(NN,test_N,test_x_bias_all,test_y_onehot_all)
        train_J_list.append(train_Javg)
        test_J_list.append(test_Javg)
 
    # prediction for training data
    train_pred = []
    for i in range(train_N):
        train_x = np.expand_dims(train_x_bias_all[i],axis=1)        # one example
        train_y = np.expand_dims(train_y_onehot_all[i],axis=1)      # one example
        NN.forward(train_x,train_y)
        train_pred.append(np.argmax(NN.y_hat))

    # prediction for test data
    test_pred = []
    for i in range(test_N):
        test_x = np.expand_dims(test_x_bias_all[i],axis=1)        # one example
        test_y = np.expand_dims(test_y_onehot_all[i],axis=1)      # one example
        NN.forward(test_x,test_y)
        test_pred.append(np.argmax(NN.y_hat))
    
    train_error = np.count_nonzero(train_y_all!=train_pred)/train_N
    test_error = np.count_nonzero(test_y_all!=test_pred)/test_N

    output_label(train_output, train_pred)
    output_label(test_output, test_pred)
    output_metrics(metrics_out,train_J_list,test_J_list,train_error,test_error)
    # print("train_error",train_error)
    # print("test_error",test_error)

    # Plot Loss
    plt.plot(range(num_epoch),train_J_list,label="Training")
    plt.plot(range(num_epoch),test_J_list,label="Test")
    plt.legend()
    plt.title("Average Cross-Entropy vs. Epoch - Learning Rate = 0.001")
    plt.xlabel("Epoch")
    plt.ylabel("Average Cross-Entropy")
    plt.show()