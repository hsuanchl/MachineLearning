import numpy as np
import sys
from collections import OrderedDict

#extracts text from file and returns a list with label and separated words
def load_tsv(path):
    lines = []  #save in to list of lines
    file = open(path, "r") 
    for line in file:
        line = line.rstrip('\n').rstrip(' ')    #remove newline and space from the end of sentence
        line = line.split('\t')                 #separate the label from text
        line[1] = line[1].split(' ')            #separate words from sentence
        lines.append(line)                      #line now has 2 lists. line[0] = label, line[1] = list of words
    file.close()
    return lines

#returns dictionary with words as keys and the index as values
def load_dict(dict_input):                      #extracts text from file and returns a dictionary
    dictionary = {}
    file = open(dict_input, "r") 
    for line in file:
        line = line.rstrip('\n').rstrip(' ')
        line = line.split(' ')
        dictionary[line[0]]=line[1]             #insert key and value
    file.close()
    return dictionary

#returns formatted representation of the words - convert word to the index of the word in dictionary (formatted_words[i][j]: ith line, jth word)
#value is 1 if word appears in sentence, otherwise 0
def model1(dictionary,data):    
    formatted_words = []
    for line in data:
        formatted_words_line = []           
        words = line[1]
        for word in words:
            if word in dictionary:
                formatted_words_line.append(dictionary[word])
        formatted_words_line = list(OrderedDict.fromkeys(formatted_words_line))               #remove duplicates while keeping order
        formatted_words.append(formatted_words_line)
    return formatted_words

#returns formatted representation of the words - convert word to the index of the word in dictionary (formatted_words[i][j]: ith line, jth word)
#value is 1 if word appears in sentence and word occurance < 4 times,  otherwise 0
def model2(dictionary,data):    
    trim_thresh = 4
    formatted_words = []
    for line in data:
        formatted_words_line = []           
        words = line[1]
        count = {}                  #count number of occurances of each word in a line
        erase = []                  #if too many occurances, we put the element into the erase list
        for word in words:
            if word in dictionary:
                formatted_words_line.append(dictionary[word])
                if dictionary[word] in count:
                    count[dictionary[word]] += 1
                else:
                    count[dictionary[word]] = 1
                if count[dictionary[word]] == trim_thresh:                  
                    erase.append(dictionary[word])
        formatted_words_line = list(OrderedDict.fromkeys(formatted_words_line))         #remove duplicates while keeping order
        formatted_words_line = [w for w in formatted_words_line if w not in erase]      #delete all elements in erase list
        formatted_words.append(formatted_words_line)
    return formatted_words

#select model
def model(feature_flag,dictionary,data):
    if feature_flag == '1':
        return model1(dictionary,data)
    elif feature_flag == '2':
        return model2(dictionary,data)
    else:
        print("WRONG FEATURE_FLAG NUMBER!")

# save formatted words in sparse dictionary form to file
def save_formatted_data(output_path,data,formatted_words):
    file = open(output_path,"w")
    for label,line in zip(data,formatted_words):
        file.write(label[0])
        for fw in line:
            file.write('\t')
            file.write(fw)
            file.write(":1")
        file.write("\n")
    file.close()

if __name__== "__main__":
    #arguments
    train_input = sys.argv[1]
    validation_input = sys.argv[2]
    test_input = sys.argv[3]
    dict_input = sys.argv[4]
    formatted_train_out = sys.argv[5]
    formatted_validation_out = sys.argv[6]
    formatted_test_out = sys.argv[7]
    feature_flag = sys.argv[8]

    #load data
    dictionary = load_dict(dict_input)
    train_data = load_tsv(train_input)
    validation_data = load_tsv(validation_input)
    test_data = load_tsv(test_input)

    #format data
    train_formatted_words = model(feature_flag,dictionary,train_data)
    validation_formatted_words = model(feature_flag,dictionary,validation_data)
    test_formatted_words = model(feature_flag,dictionary,test_data)

    #save data
    save_formatted_data(formatted_train_out,train_data,train_formatted_words)
    save_formatted_data(formatted_validation_out,validation_data,validation_formatted_words)
    save_formatted_data(formatted_test_out,test_data,test_formatted_words)