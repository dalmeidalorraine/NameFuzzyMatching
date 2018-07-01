# Python scripts for Fuzzy Name Matching problem for ING

Scripts that run on the command line, i.e. *Terminal* on Mac OS X, and *Command Prompt* on Windows.

The codes are built using Python 3.6.4 on Anaconda

## Description

### `executeNameMatching.py`
This script takes the external source dataset test file as the input and calls the class which executes the name matching against the ground truth of names.

Use the below command to execute the script:
`python executeNameMatching.py -i filename`
`-i`: argument for input filename

It creates an output file `result.csv` in the execution directory with columns
    * test_index: index of the company in an external source dataset 
    * company_id: the predicted match of this entry to G.

Command line outputs: `Time for execution` which gives the execution time in seconds

#### Dependencies
`ground_truth.npz` - tf-idf matrix of the ground truth dataset during training
`vectorizer.pickle` - vectorizer used to transform the ground truth dataset
---

### `nameMatching.py`

Defines the class `FuzzyNameMatching` which includes the below steps:
*	Slice the external source dataset into slices of 10,000 names each
*	Pre-process the external source dataset and converts to tf-idf sparse matrix
*	Compute dot product of test set tf-idf matrix with ground truth tf-idf matrix
*	Select top one similarity result from the resulting matrix
*	Save the final result as csv file
---


