
import sys
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier

df = pd.read_csv("Churn_Modelling.csv")
del df['RowNumber']
del df['Surname']
del df['Geography']
del df['CustomerId']
df['Gender'] = df['Gender'].replace({'Female': 1, 'Male': 0})
df['Cinsiyet'] = df['Gender']
del df['Gender']

#: SHUFFLE
df = df.sample(frac = 1.0)

columns = list(df.columns)
columns.remove('Exited')

tum_y = df['Exited']
tum_x = df[ columns ]


#: SPLIT INTO TRAIN TEST
number_of_rows = len(df) # 10000
train_count = int(number_of_rows * 0.70)

train = df[:train_count]
test = df[train_count:]

train_y = train['Exited']
train_x = train[ columns ]

test_y = test['Exited']
test_x = test[ columns ]


#: CREATE THE CLASSIFIERS

algorithms = [
    RandomForestClassifier(), 
    AdaBoostClassifier(), 
    GaussianNB(), 
    svm.SVC(), 
    #NearestNeighbors(), 
    MLPClassifier(random_state=1, max_iter=300) 
]

# 1600 tane 


"""
for a in algorithms:
    a.fit( train_x, train_y )
    pred = a.predict( test_x )
    print( a, accuracy_score( test_y, pred ) )

"""

clf = RandomForestClassifier(max_depth=5, min_samples_leaf=10)
clf.fit( train_x, train_y )



estimator = clf.estimators_[5]

from sklearn.tree import export_graphviz
# Export as dot file
export_graphviz(estimator, out_file='tree.dot', 
                feature_names = columns,
                class_names = ['gidecek', 'kalacak'],
                rounded = True, proportion = False, 
                precision = 2, filled = True)

# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# Display in jupyter notebook
#from IPython.display import Image
#Image(filename = 'tree.png')





sys.exit(1)







