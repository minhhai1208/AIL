import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pickle
from sklearn.linear_model import LogisticRegression
columns = [
       'Question 2: Sound (1-5)', 'Question 2.1: Music (1-5)',
       'Question 2.2 : Noise control (1-5)',
       'Question 2.3 : Speaking Style (1-5)', 'Question 3: Camera (1-5)',
       'Question 3.1: Stable (1-5)', 'Question 3.2: Angel diversity (0-1)',
       'Question 4: Images (1-5)', 'Question 4.1: Resolution (1-5)',
       'Question 4.2: Color (1-5)', 'Question 5: Content (1-5)',
       'Question 5.1: Introduction (0-1)',
       'Question 5.2: Food description (0-1)', 'Question 6: Reviewer (1-5)',
       'Question 6.1: Reviewer emotion is negative - neutral - positive (1-3)',
       'Question 6.2: Recommendation (0-1)',
       'Question 6.3: Clear information (0-1)']
core_columns = [
       'Question 2: Sound (1-5)','Question 3: Camera (1-5)',
       'Question 4: Images (1-5)', 'Question 5: Content (1-5)',
       'Question 6: Reviewer (1-5)',]
output_column = 'Attractive Level (1-5)'

data = pd.read_excel("C:\\Users\\user\\Downloads\\mean_data.xlsx")
df = pd.DataFrame(set(data['video id']), columns=['video id'])
for id in df['video id']:
    c = columns + [output_column]
    mean = data[data['video id'] == id][c].mean().round()
    df.loc[df['video id'] == id, c] = np.array(mean)

X_train, X_test, y_train, y_test = train_test_split(data[core_columns][:], data[output_column][:], test_size=0.2, shuffle=True)

def LoR():
    clf = LogisticRegression(C=1, solver="lbfgs", max_iter=1e4)
    y1 = y_train.copy()
    y1[y1 < 3] = 0
    y1[y1 >= 3] = 1
    y2 = y_test.copy()
    y2[y2 < 3] = 0
    y2[y2 >= 3] = 1
    clf.fit(X_train, y1)
    pickle.dump(clf, open('./Logistic_Regression_model', 'wb'))
    return (f1_score(clf.predict(X_train), y1, average='macro'), f1_score(clf.predict(X_test), y2, average='macro'))

result = LoR()
print('Logistic Regression:')
print('train f1: ', result[0])
print('test f1: ', result[1])