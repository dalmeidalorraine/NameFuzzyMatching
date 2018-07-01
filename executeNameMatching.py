#!/usr/bin/python
__author__ = "Lorraine DAlmeida"
__version__ = "1.0"
__email__ = "dalmeida.lorraine@gmail.com"

import sys, getopt
import pandas as pd
from scipy import sparse
import re
from nameMatching import FuzzyNameMatching
from time import time

def ngrams(string, n=3):
        string = re.sub(r'[,-./\']',r'', string)
        ngrams = zip(*[string[i:] for i in range(n)])
        return [''.join(ngram) for ngram in ngrams]

def main(argv):
   init_time = time()
   inputfile = ''
   try:
      opts, args = getopt.getopt(argv,"hi:",["ifile="])
   except getopt.GetoptError:
      print('usage: executeNameMatching.py -i <inputfile>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('usage: executeNameMatching.py -i <inputfile>')
         sys.exit()
      if opt in ("-i", "--ifile"):
         inputfile = arg
   try:
      ground_truth = pd.read_csv("Datasets/G.csv", sep='|')
      s_test = pd.read_csv(inputfile, sep='|')
      ground_truth_matrix = sparse.load_npz("ground_truth.npz")
   except Exception as e:
      print(e)
   ing_name_matching = FuzzyNameMatching(ground_truth, s_test, ground_truth_matrix)
   ing_name_matching.run()
   final_time = time() - init_time
   print("Time for execution: {}s".format(final_time))

if __name__ == "__main__":
   main(sys.argv[1:])