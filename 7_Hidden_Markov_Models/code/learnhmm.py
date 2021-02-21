import sys
import numpy as np
import logging
logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)


def load_data(path):
    data = []
    file = open(path, "r") 
    for line in file:
        line = line.rstrip('\n').rstrip(' ')
        line = line.replace('_', ' ')   #can also be done using import re
        line = line.split(' ')
        data.append(line)
    file.close()
    states = []
    words = []
    for example in data:
        states.append(example[1::2])    #even columns are states
        words.append(example[::2])      #odd columns are words
    return states, words

 #create dictionary corresponding word/tag to index
def load_index(index_path):            
    index_dic = {}
    i = 1
    file = open(index_path, "r") 
    for line in file:
        line = line.rstrip('\n').rstrip(' ')
        index_dic[line] = i
        i += 1
    file.close()
    return index_dic

def output_matrix(output_path, output_data):
    file = open(output_path,"w")
    for row in output_data:
        for x in row:
            file.write(str(x))
            file.write(" ")
        file.write("\n")
    file.close()



if __name__ == "__main__":
    #arguments
    train_input = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmmprior = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]

    #load data and index
    train_states, train_words = load_data(train_input)

    #for evaluating log likelihood using partial data
    if False:
        train_states = train_states[:10000]
        train_words = train_words[:10000]

    tag_index_dic = load_index(index_to_tag)
    word_index_dic = load_index(index_to_word)

    num_states = len(tag_index_dic)
    num_words = len(word_index_dic)
    num_ex = len(train_states)

    #convert words and tags to index
    for i in range(len(train_states)):
        for j in range(len(train_states[i])):
            train_states[i][j] = tag_index_dic[train_states[i][j]]
        for j in range(len(train_words[i])):
            train_words[i][j] = word_index_dic[train_words[i][j]]

    # initialization probabilities P(y1 = j) = prior_j
    N_prior = np.zeros(num_states)             #N_prior[j] is the # of times state j+1 is associated with the first word of a sentence in the training data set 
    for example in train_states:
        N_prior[example[0]-1] += 1             #-1 because index of words and tags start with 1

    prior = np.expand_dims((N_prior+1)/np.sum(N_prior+1),axis=1)            #add pseudocount of 1
    logging.debug("prior\n %s" %prior)   

    #transition probabilities P(yt=k|yt-1 = j) = a_jk
    N_A = np.zeros((num_states,num_states))
    for example in train_states:
        for i in range(len(example)-1):
            N_A[example[i]-1,example[i+1]-1] += 1
    A = (N_A+1)/np.expand_dims(np.sum(N_A+1,axis = 1),axis=1)      #divided by sum over rows
    logging.debug("transtion\n %s" %A)

    #emission probabilities P(xt=k|yt = j) = b_jk
    N_B = np.zeros((num_states,num_words))
    for tags,words in zip(train_states,train_words):
        for i in range(len(tags)):
            N_B[tags[i]-1,words[i]-1] += 1
    B = (N_B+1)/np.expand_dims(np.sum(N_B+1,axis = 1),axis=1)      #divided by sum over rows
    logging.debug("emission\n %s" %B)

    #output to file
    output_matrix(hmmprior, prior)
    output_matrix(hmmtrans, A)
    output_matrix(hmmemit, B)