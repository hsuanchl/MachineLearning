# Neural Network
Label images of handwritten letters with Neural Network.

The objective function is the average cross entropy over the training dataset.

## Usage
    python neuralnet.py <train input> test input> <train out> <test out> <metrics out> <num epoch> <hidden units> <init flag> <learning rate>

1. `<train input>`: path to the training input .csv file
2. `<test input>`: path to the test input .csv file
3. `<train out>`: path to output .labels file to which the prediction on the training data should be written
4. `<test out>`: path to output .labels file to which the prediction on the test data should be written
5. `<metrics out>:` path of the output .txt file to which metrics such as train and test error should be written
6. `<num epoch>`: integer specifying the number of times backpropogation loops through all of the training data
7. `<hidden units>`: positive integer specifying the number of hidden units
8. `<init flag>`: integer taking value 1 or 2 that specifies whether to use RANDOM or ZERO initialization
9. `<learning rate>`: float value specifying the learning rate for SGD

## Example
Predict handwritten letters

    python neuralnet.py largeTrain.csv largeTest.csv model1train_out.labels model1test_out.labels model1metrics_out.txt 10 50 1 0.01

## Data
OCR dataset: 16 x 8 image in row major format