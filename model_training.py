#%%
# Import libraries
import datetime
from importlib import reload
import pandas as pd
import numpy as np
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn.metrics import accuracy_score
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_selection import RFECV
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn import feature_selection
from sklearn import metrics
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings('ignore')
import data_cleaning
reload(data_cleaning)
from data_cleaning import get_data
#%%
# Get data and labels
years = list(range(1999, 2021))
_data = get_data(years, None, 5)
#%%
data = _data.copy()
X_full = data["X"]
Y_full = data["Y"]

X_train = X_full.loc[X_full.season != 2020, ~X_full.columns.isin(['game_id'])]
Y_train = Y_full.loc[Y_full.season != 2020, "result"]

X_test = X_full.loc[X_full.season == 2020, ~X_full.columns.isin(['game_id'])]
Y_test = Y_full.loc[Y_full.season == 2020, "result"]
# %%
# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Split training data into train and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42) # 80% training and 20% test

# Initialize the DMatrix objects
dtrain = xgb.DMatrix(X_train, label=Y_train)
dval = xgb.DMatrix(X_val, label=Y_val)
dtest = xgb.DMatrix(X_test, label=Y_test)

# Set the parameters for xgboost
param = {
    'max_depth': 5,  # the maximum depth of each tree
    'eta': 0.3,  # the training step for each iteration
    'objective': 'binary:logistic',  # error evaluation for multiclass training
    'eval_metric': 'logloss',  # evaluation metric
    'min_child_weight': 1,
    'gamma': 0.2,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 1,
    'alpha': 0.5,
    'lambda': 0.5,
}  # the number of classes that exist in this datset
num_round = 100000  # the number of training iterations

# Do cross validation
cv_results = xgb.cv(
    param,
    dtrain,
    num_boost_round=num_round,
    nfold=5,
    metrics={'logloss'},
    early_stopping_rounds=100,
    seed=0,
)
# Plot cross validation results with matplotlib
import matplotlib.pyplot as plt
plt.plot(cv_results['train-logloss-mean'], label='train logloss')
plt.plot(cv_results['test-logloss-mean'], label='test logloss')
plt.xlabel('num_round')
plt.ylabel('logloss')
plt.legend()
plt.show()

#%%
# Train the model with eval metric
bst = xgb.train(param, dtrain, num_round, evals=[(dval, "Validation")])

# Make predictions on the test set
preds = bst.predict(dtest)

# Evaluate the accuracy of the model
accuracy = accuracy_score(Y_test, preds.round())
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Evaluate ROC/AUC of the model
fpr, tpr, thresholds = metrics.roc_curve(Y_test, preds)
auc = metrics.roc_auc_score(Y_test, preds)
print("AUC: %.2f%%" % (auc * 100.0))

# Plot ROC/AUC of the model
import matplotlib.pyplot as plt
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()
# %%
#%%
# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a list of standard classifiers
classifiers = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    #Gaussian Processes
    gaussian_process.GaussianProcessClassifier(),

    #GLM
    linear_model.LogisticRegressionCV(),

    #Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),

    #SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),

    #Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),

    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html
#     XGBClassifier()    
]

# Define a functiom which finds the best algorithms for our modelling task
def find_best_algorithms(classifier_list, X, y):
    # This function is adapted from https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling
    # Cross validate model with Kfold stratified cross validation
    kfold = StratifiedKFold(n_splits=5)

    # Grab the cross validation scores for each algorithm
    cv_results = [cross_val_score(classifier, X, y, scoring = "neg_log_loss", cv = kfold) for classifier in classifier_list]
    cv_means = [cv_result.mean() * -1 for cv_result in cv_results]
    cv_std = [cv_result.std() for cv_result in cv_results]
    algorithm_names = [alg.__class__.__name__ for alg in classifiers]

    # Create a DataFrame of all the CV results
    cv_results = pd.DataFrame({
        "Mean Log Loss": cv_means,
        "Log Loss Std": cv_std,
        "Algorithm": algorithm_names
    })

    return cv_results.sort_values(by='Mean Log Loss').reset_index(drop=True)

best_algos = find_best_algorithms(classifiers, X_train, Y_train)
best_algos

# Try a logistic regression model and see how it performs in terms of accuracy
kfold = StratifiedKFold(n_splits=5)
cv_scores = cross_val_score(LogisticRegressionCV(), X_train, Y_train, scoring='accuracy', cv=kfold)
cv_scores.mean()

# Try a XGBoostClassifier model and see how it performs in terms of accuracy
kfold = StratifiedKFold(n_splits=5)
cv_scores = cross_val_score(XGBClassifier(), X_train, Y_train, scoring='accuracy', cv=kfold)
cv_scores.mean()
