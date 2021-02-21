# Decision Stump
Perform majority vote on dataset.

## Usage
    python3 decisionStump.py <train input> <test input> <split index> <train out> <test out> <metrics out>
1. `<train input>`: path to the training input .tsv file
2. `<test input>`: path to the test input .tsv file
3. `<split index>`: the index of feature at which we split the dataset. The first column has index 0, the second column index 1, and so on
4. `<train out>`: path of output .labels file to which the predictions on the training data should be written
5. `<test out>`: path of output .labels file to which the predictions on the test data should be written
6. `<metrics out>`: path of the output .txt file to which metrics such as train and test error should be written

## Example
Predict whether a US politician is a member of the Democrat or Republican party, based on their past voting history.

    python3 decisionStump.py politicians_train.tsv politicians_test.tsv 0 pol_0_train.labels pol_0_test.labels pol_0_metrics.txt
