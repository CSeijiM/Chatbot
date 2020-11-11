import pandas as pd
import numpy as np
from sklearn import tree

train = pd.read_csv("./zipcode98065.csv")
#train['CARO'] = train.CARO.apply(lambda x : 1 if x is True else 0)
train = train.drop('date',axis=1)
train = train.drop('id',axis=1)
train = train.drop('preco_por_area',axis=1)
train = train.drop('CARO',axis=1)
train = train.drop('waterfront',axis=1)
train = train.drop('view',axis=1)
train = train.drop('yr_renovated',axis=1)
train = train.drop('lat',axis=1)
train = train.drop('long',axis=1)
train = train.drop('sqft_living15',axis=1)
train = train.drop('sqft_lot15',axis=1)
train = train.drop('sqft_above',axis=1)
train = train.drop('sqft_basement',axis=1)
train = train.iloc[0:300, 0:15]

y_train = train['CARO_B']
x_train = train.drop(['CARO_B'], axis=1).values
decision_tree = tree.DecisionTreeClassifier(max_depth = 5)
print(train)
decision_tree.fit(x_train, y_train)

with open("res.dot", 'w') as f:
    f = tree.export_graphviz(decision_tree,
                            out_file=f,
                            max_depth=20,
                            impurity=True,
                            feature_names=list(train.drop(['CARO_B'], axis=1)),
                            class_names = ['False', 'True'],
                            rounded=True,
                            filled=True )