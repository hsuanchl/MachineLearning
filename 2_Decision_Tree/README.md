# Decision Tree
Binary classifier with a Decision Tree learner.

`inspection.py` 
calculates the overall Gini impurity (i.e. the Gini impurity of the labels for the entire dataset and before any splits) and the error rate (the percent of incorrectly classified instances) of classifying using a majority vote 

`decisionTree.py`
learns a decision tree with a specified maximum depth, prints the decision tree in a specified format, predicts the labels of the training and testing examples, and calculates training and testing errors.

## Usage
    python3 inspection.py <input> <output>

    python3 decisionTree.py <train input> <test input> <max depth> <train out> <test out> <metrics out>
1. `<train input>`: path to the training input .tsv file
2. `<test input>`: path to the test input .tsv file 
3. `<max depth>`: maximum depth to which the tree should be built
4. `<train out>`: path of output .labels file to which the predictions on the training data should be written
5. `<test out>`: path of output .labels file to which the predictions on the test data should be written
6. `<metrics out>`: path of the ouput.txt file to which metrics such as train and test error should be written

## Example
Predict whether a US politician is a member of the Democrat or Republican party, based on their past voting history.

    python3 inspection.py small_train.tsv small_inspect.txt3
    
    python3 decisionTree.py politicians_train.tsv politicians_test.tsv 3 pol_3_train.labels pol_3_test.labels pol_3_metrics.txt