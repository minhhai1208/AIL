import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pickle

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

print(df)

X_train, X_test, y_train, y_test = train_test_split(data[core_columns][:], data[output_column][:], test_size=0.2, shuffle=True)
clf = SVC(C=2)
clf.fit(X_train, y_train)
pickle.dump(clf, open('./SVM_model', 'wb'))
print('Support Vector Machine:')

print('train accuracy: ',f1_score(clf.predict(X_train), y_train, average='macro'))
print('test accuracy: ',f1_score(clf.predict(X_test), y_test, average='macro'))


oversample = SMOTE(k_neighbors=2, random_state=0)
X, y = oversample.fit_resample(data[core_columns][:],data[output_column][:])
# X, y = pf[core_columns][:], pf[output_column][:]
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2, shuffle=True)

print(X_train1.shape)
from sklearn.svm import SVC
clf = SVC(C=2)
clf.fit(X_train1, y_train1)
pickle.dump(clf, open('./SVM_model', 'wb'))
print('Support Vector Machine:')

print('train accuracy: ',f1_score(clf.predict(X_train1), y_train1, average='macro'))
print('test accuracy: ',f1_score(clf.predict(X_test1), y_test1, average='macro'))