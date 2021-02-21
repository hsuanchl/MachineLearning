# Decision Tree
NLP sentiment polarity analyzer with binary logistic regression.

`feature.py` extracts features from raw data and formats training, validation, and test data using the dictionary.

`lr.py` learns parameters of a binary logistic regression model that predicts a sentient polarity for the correspnoding feature vector.

## Usage
    python feature.py <train input> <validation input> <test input> <dict input> <formatted train> <formatted validation> <formatted test> <feature flag>

    python lr.py <formatted train> <formatted validation> <formatted test> <dict input> <train out> <test out> <metrics out> <num epoch>

1. `<train input>`: path to the training input .tsv file
2. `<validation input>`: path to the validation input .tsv file
3. `<test input>`: path to the test input .tsv file
4. `<dict input>`: path to the dictionary input .txt file
5. `<formatted train>`: path to output .tsv file to which the feature extractions on the training data should be written
6. `<formatted validation>`: path to output .tsv file to which the feature extractions on the validation data should be written
7. `<formatted test>`: path to output .tsv file to which the feature extractions on the test data should be written
8. `<feature flag>`: integer taking value 1 or 2 that specifies whether to construct the Model 1 feature set or the Model 2 feature set
9. `<train out>`: path to output .labels file to which the prediction on the training data should be written 
10. `<test out>`: path to output .labels file to which the prediction on the test data should be written
11. `<metrics out>`: path of the output .txt file to which metrics such as train and test error should be written 
12. `<num epoch>`: integer specifying the number of times SGD loops through all of the training data

## Example
Determine whether movie review is positive or negative.

    python feature.py train_data.tsv valid_data.tsv test_data.tsv dict.txt formatted_train.tsv formatted_valid.tsv formatted_test.tsv 1

    python lr.py formatted_train.tsv formatted_valid.tsv formatted_test.tsv dict.txt train_out.labels test_out.labels metrics_out.txt 60

## Data
Dataset: Label is 0 for negative review and 1 for positive review. Attribute is the review text.

Dictionary: Vocabulary including words from training data