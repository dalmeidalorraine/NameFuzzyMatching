#!/usr/bin/python
__author__ = "Lorraine DAlmeida"
__version__ = "1.0"
__email__ = "dalmeida.lorraine@gmail.com"

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

class FuzzyNameMatching:
    def __init__(self, gt, st, gt_mat):
        self.groundTruth = gt
        self.sTest = st
        self.ground_truth_matrix = gt_mat

    # Function to return top results of cosine similarities from the dot product
    def get_top_sim(self, sparse_row):
        nnz = sparse_row.getnnz()
        if nnz==0:
            return (0.0, None, -1)
        else:
            arg_index = np.argpartition(sparse_row.data, -1)[-1]
            match_id = sparse_row.indices[arg_index]
            match_score = sparse_row.data[arg_index]
            if match_score<0.60:
                result = (0.0, "NaN", -1)
            else:
                result = (match_score, self.groundTruth.loc[match_id]['name'], 
                         self.groundTruth.loc[match_id]['company_id'])
        return result

    # Function to compute matrix dot product
    def cosine_similarities(self, trainMat, groundTruthMat):
        sim = trainMat.dot(groundTruthMat.T)
        #sim = trainMat*groundTruthMat.T.tocsc()
        return [self.get_top_sim(row) for row in sim]

    # Function call to cosine similarity computation
    def execute_matching(self, sTestFil, match_df, vectorizer):
        test_matrix = vectorizer.transform(sTestFil['name'])
        res = self.cosine_similarities(test_matrix, self.ground_truth_matrix)
        match_score, match_name , match_company_id = zip(*res)
        sTestFil['match_company_id'] = np.array(match_company_id)
        match_df = match_df.append(sTestFil[["test_index", "match_company_id"]])
        return match_df

    def index_range(self, nrows, chunk_size):
        return range(1 * chunk_size, (nrows // chunk_size ) * chunk_size, chunk_size)

    # Function to split external test dataset into chunks
    def split(self, dfm, chunk_size):
        indices = self.index_range(dfm.shape[0], chunk_size)
        return np.split(dfm, indices)

    def run(self):
        try:
            vectorizer = pickle.load(open("vectorizer.pickle", "rb"))
        except Exception as e:
            print(e)
        slices = self.split(self.sTest, 10000)
        df_ = pd.DataFrame(columns=["test_index", "match_company_id"])
        for sTestFil in slices:
            df_ = self.execute_matching(sTestFil, df_, vectorizer)
        df_.to_csv("result.csv", sep='|', header=True, index=False)

