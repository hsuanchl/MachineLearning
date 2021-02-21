# Hidden Markov Model

`learnhmm.py` learns the hidden Markov model parameters (initialization probabilities, transition probabilities, and emission probabilities) needed to apply the forward backward algorithm. Probabilities are modeled using a multinomial distribution and estimated with maximum likelihood. 

`forwardbackward.py` uses the forward-backward algorithm to estimate the conditional probabilities and assigns tags using the minimum Bayes risk predictor.


## Usage
    python learnhmm.py <train input> <index to word> <index to tag> <hmmprior> <hmmemit> <hmmtrans>

    python forwardbackward.py <test input> <index to word> <index to tag> <hmmprior> <hmmemit> <hmmtrans> <predicted file> <metric file>

1. `<train input>`: path to the training input .txt file
2. `<index to word>`: path to the .txt that specifies the dictionary mapping from words to indices
3. `<index to tag>`: path to the .txt that specifies the dictionary mapping from tags to indices
4. `<hmmprior>`: path to output .txt file to which the estimated prior will be written
5. `<hmmemit>`: path to output .txt file to which the emission probabilities will be written
6. `<hmmtrans>`: path to output .txt file to which the transition probabilitiess will be written
7. `<test input>`: path to the test input .txt file that will be evaluated by your forward backward
algorithm
6. `<predicted file>`: path to the output .txt file to which the predicted tags will be written
7. `<metric file>`: path to the output .txt file to which the metrics will be written


## Example
Predicts tags for words in sentences.

    python learnhmm.py trainwords.txt index_to_word.txt index_to_tag.txt hmmprior.txt hmmemit.txt hmmtrans.txt 

    python forwardbackward.py testwords.txt index_to_word.txt index_to_tag.txt hmmprior.txt hmmemit.txt hmmtrans.txt predicttest.txt metrics.txt 

## Data
Training and test datasets include text. Each word is labeled with a tag.

`trainwords.txt` contains labeled text data

`index to word.txt` and `index to tag.txt` contain a list of all words or tags that appear in the data set.