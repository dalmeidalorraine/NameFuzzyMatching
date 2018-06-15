# NameFuzzyMatching

All data is provided as plain text column separated files:

Column separated by ‘|’ (bar), one line per entry;
the test set will have the same size as STrain provided to you now (hence, no memory issues during test). We provide you up-front with three datasets: G, STrain and sample_submission
G has as columns:
company_id: the id of the company in our ground-truth administration
name: the name of the company in our ground-truth administration
STrain has as columns:
train_index: an index of the company in the external source dataset;
name: the name of the company as represented in the external source dataset;
company_id: the correct match of this entry to G. Is -1 if correct label is ‘not in G’, otherwise corresponds to G_id
sample_submission has as columns:
test_index: index of the company in an external source dataset (note: different index from STrain, full STest will only be provided during the interview);
company_id: the predicted match of this entry to G. Note that in this file, these are randomly generated predictions.
You need to design, code and train a model that predicts company_id, minimizing the cost function as specified on the previous page Please make sure that your trained model:

accepts as input a plain text file of the format STest, containing two columns test_index and name;
runs from the command line, using as input the path to the file STest;
prediction time should be ‘near real time’, i.e. about 1 minute for 10,000 entries (on a regular laptop);
It should return a file with the above plain text format, including the columns test_index and company_id (an example submission is provided).
