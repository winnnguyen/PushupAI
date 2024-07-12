import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_csv('landmarks.csv')
X = df.drop(['Unnamed: 0', '0'], axis=1)
y = df['0']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train.values, y_train.values)

with open('model_pickle', 'wb') as f:
    pickle.dump(model, f)