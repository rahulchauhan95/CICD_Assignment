import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np
from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC

df = pd.read_csv("data/train.csv")
X = df.drop(columns=['Disease']).to_numpy()
y = df['Disease'].to_numpy()
labels = np.sort(np.unique(y))
y = np.array([np.where(labels == x) for x in y]).flatten()

# Create and train SVM model (0.457)
model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42).fit(X, y)

# Create and train Naive Bayes model (0.523)
# model = GaussianNB().fit(X, y)

with open("model.pkl", 'wb') as f:
    pickle.dump(model, f)