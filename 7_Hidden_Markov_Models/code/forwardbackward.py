import sys
import numpy as np
from learnhmm import load_data, load_index
from copy import deepcopy

def load_matrix(path):
    data = []
    file = open(path, "r") 
    for line in file:
        line = line.rstrip('\n').rstrip(' ')
        line = line.split(' ')
        line = list(map(float, line))
        data.append(line)
    file.close()
    data = np.vstack((data))
    return data

def output_label(output_path, test_words_orig, tag_list):
    file = open(output_path,"w")
    for words, tags in zip(test_words_orig,tag_list):
        for i in range(len(words)):
            file.write(str(words[i]))
            file.write("_")
            file.write(tags[i])
            if i != len(words)-1:
                file.write(" ")
        file.write("\n")
    file.close()

def output_metrics(output_path, final_avg_log_likelihood, accuracy):
    file = open(output_path,"w")
    file.write("Average Log-Likelihood: ")
    file.write(str(final_avg_log_likelihood))
    file.write("\n")
    file.write("Accuracy: ")
    file.write(str(accuracy))
    file.close()

#forward pass: alpha_t(k) = p(x1...xt, yt=k)
def forward_pass(words,prior,A,B):
    T = len(words)
    alpha = np.zeros((T,num_states))        #shape is (number of words in sentence,T)x(num of states)
    for i in range(T):
        if i == 0:
            alpha[0,:] = B[:,words[0]-1] * prior.squeeze()
        else:
            alpha[i,:] = B[:,words[i]-1] * (A.T @ np.expand_dims(alpha[i-1,:],axis=1)).squeeze()
    return alpha
    
def forward_pass_logsumexp(words,prior,A,B):
    T = len(words)
    J = A.shape[0]
    # alpha = np.zeros((T,num_states))      #shape is (number of words in sentence,T)x(num of states)
    logalpha = np.zeros((T,num_states))     #shape is (number of words in sentence,T)x(num of states)
    for i in range(T):
        if i == 0:
            # alpha[0,:] = (B[:,words[0]-1] * prior.squeeze())
            logalpha[0,:] = np.log(B[:,words[0]-1]) + np.log(prior.squeeze())
        else:
            for j in range(J):
                # alpha[i,j] = B[j,words[i]-1] * np.sum(A[:,j] * alpha[i-1,:])
                v = np.log(B[j,words[i]-1]) + np.log(A[:,j]) + (logalpha[i-1,:])
                m = np.max(v)
                logalpha[i,j] = m + np.log(np.sum(np.exp(v-m)))
    return logalpha

#backward pass: beta_t(k) = p(xt+1...xT | yT=k)
def backward_pass(states,words,prior,A,B):
    T = len(words)
    beta = np.zeros((T,num_states))     #shape is (number of words in sentence,T)x(num of states)
    for i in reversed(range(T)):
        if i==T-1:
            beta[T-1,:] = np.ones(num_states)
        else:
            beta[i,:] = A @ (B[:,words[i+1]-1] * beta[i+1,:])
    return beta

def convert_index_to_tag(tag_index_dic, predicted_list):
    tag_list = []
    rev_tag_index_dic = { v:k for k,v in tag_index_dic.items()}
    for predicted_sentence in predicted_list:
        tag_list.append([rev_tag_index_dic.get(word)  for word in predicted_sentence])
    return tag_list

def accuracy(tag_list,test_states_orig):
    correctly_predicted = 0
    tag_list_flatten = [y for x in tag_list for y in x]                     # flatten list
    test_states_orig_flatten = [y for x in test_states_orig for y in x]
    for pred,orig in zip(tag_list_flatten,test_states_orig_flatten):    
        correctly_predicted += 1 if pred == orig else 0
    accuracy = correctly_predicted/len(test_states_orig_flatten)
    return accuracy

if __name__ == "__main__":
    #arguments
    test_input = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmmprior = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]
    predicted_file = sys.argv[7]
    metrics_file = sys.argv[8]

    #load data and index
    test_states_orig, test_words_orig = load_data(test_input)
    tag_index_dic = load_index(index_to_tag)
    word_index_dic = load_index(index_to_word)
    num_states = len(tag_index_dic)
    num_words = len(word_index_dic)
    num_ex = len(test_states_orig)

    #convert words and tags to index
    test_states = deepcopy(test_states_orig)
    test_words = deepcopy(test_words_orig)
    for i in range(len(test_states)):
        for j in range(len(test_states[i])):
            test_states[i][j] = tag_index_dic[test_states[i][j]]
        for j in range(len(test_words[i])):
            test_words[i][j] = word_index_dic[test_words[i][j]]

    #load HMM parameters
    prior = load_matrix(hmmprior)
    A = load_matrix(hmmtrans)           #shape is (num of states)x(num of states)
    B = load_matrix(hmmemit)            #shape is (num of states)x(num of words)
    
    #forward-backward algorithm
    predicted_list = []
    log_likelihood_list = []
    for states,words in zip(test_states,test_words):     
        alpha = forward_pass(words,prior,A,B)
        beta = backward_pass(states,words,prior,A,B)
        predicted_sentence = np.argmax(alpha*beta, axis=1)+1    #element-wise multiplication, find argmax across rows(states)
        predicted_list.append(predicted_sentence)

        #only alpha has log-sum-exp trick implemented now, should implement it on beta as well
        alpha = forward_pass_logsumexp(words,prior,A,B)
        v = alpha[-1,:]
        m = np.max(v)
        log_likelihood_list.append(m+np.log(np.sum(np.exp(v-m))))

    final_avg_log_likelihood = np.sum(log_likelihood_list)/len(log_likelihood_list)
    print("average log_likelihood: \n",final_avg_log_likelihood)

    # converted predicted index to predicted tags and write to file
    tag_list = convert_index_to_tag(tag_index_dic, predicted_list)
    output_label(predicted_file, test_words_orig, tag_list)

    # calculate accuracy and write to file
    accuracy = accuracy(tag_list,test_states_orig) 
    output_metrics(metrics_file, final_avg_log_likelihood, accuracy)
    print("accuracy: ",accuracy)

    #for 1.4, plot
    if False:
        import matplotlib.pyplot as plt
        sequences = [10,100,1000,10000]
        train_log_likelihood = [-122.54156723373114, -110.3159995426195, -101.07201479899909, -95.43736971788395]
        test_log_likelihood = [-131.6493433280383, -116.09167374707644, -105.30819271388366, -98.52825144371118]
        plt.plot(sequences,train_log_likelihood,label="Train")
        plt.plot(sequences,test_log_likelihood,label="Test")
        plt.legend()
        plt.title("Average Log-Likelihood vs. Number of Training Sequences")
        plt.xlabel("Training Sequences")
        plt.ylabel("Average Log-Likelihood")
        plt.show()