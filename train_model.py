import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import pickle

# Load dataset
data = pd.read_csv('heart.csv')

# Replace missing values
imputer = SimpleImputer(strategy="mean")
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Convert target values to binary
data_imputed['target'] = data_imputed['target'].apply(lambda x: 1 if x > 0 else 0)

# Split features and target
X = data_imputed.drop('target', axis=1)
y = data_imputed['target']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)
