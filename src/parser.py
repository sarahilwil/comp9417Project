__version__ = '0.1.0'

import numpy as np
import csv


class Parse:

    ### FIRST STRUCTURE ###
    # Dictionary of each column
    # List of article numbers
    # List of dictionaries containing word counts
    # List of the identified/correct topics
    # Training = {No. = [], words = [dicts()], topic = []}
    
    ### SECOND STRUCTURE ###
    # List (row) of List (column)
    
    ### THIRD STRUCTURE ###
    # Dictionary of article No.
    # each entry is a List storing = [ Dict{ Words }, topic ]
    
    ### FOURTH STRUCTURE ###
    # Dictionary of article No.
    # each entry is a List storing = [ [words], topic ]
    
    def __init__(self, style=1):
        if style == 1:
            self.trainSet = dict()
            style1("data/training.csv")
            self.testSet = dict()
        elif style == 2:
            self.trainSet = []
            self.testSet = []
        elif style == 3:
            self.trainSet = dict()
            self.testSet = dict()
        elif style == 4:
            self.trainSet = dict()
            self.testSet = dict()
        else:
            print("ERROR: Please give a style number")
        
    def style1(self, fileName):
        with open(fileName, newline='') as f:
            next(f)
            preformat = csv.DictReader(f, ['No', 'Words', 'Topic'])
            
            print(preformat['No'])
            print(preformat['Topic'])
        


