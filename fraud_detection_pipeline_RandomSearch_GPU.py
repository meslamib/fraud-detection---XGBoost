import pandas as pd
import xgboost as xgb
import numpy as np
from xgboost import plot_tree
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn_pandas import DataFrameMapper
from sklearn_pandas import CategoricalImputer
from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import Imputer
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import FeatureUnion
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

train_identity = pd.read_csv('train_identity.csv.zip')
train_transaction = pd.read_csv('train_transaction.csv.zip')
print(train_identity.shape)
print(train_transaction.shape)
data = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
data = data.drop('TransactionID', axis=1)
print(data.shape)



X, y = data.iloc[:,data.columns!='isFraud'],data.loc[:,data.columns=='isFraud']

print(X.shape)
print(y.shape)

# Check number of nulls in each feature column
nulls_per_column = X.isnull().sum()
print(nulls_per_column)


# Create a boolean mask for categorical columns
categorical_feature_mask = X.dtypes == object


# Get list of categorical column names
categorical_columns = X.columns[categorical_feature_mask].tolist()

# Get list of non-categorical column names
non_categorical_columns = X.columns[~categorical_feature_mask].tolist()


# Apply numeric imputer
numeric_imputation_mapper = DataFrameMapper(
                                            [([numeric_feature], SimpleImputer(strategy="median")) for numeric_feature in non_categorical_columns],
                                            input_df=True,
                                            df_out=True
                                           )

# Apply categorical imputer
categorical_imputation_mapper = DataFrameMapper(
                                                [(category_feature, CategoricalImputer()) for category_feature in categorical_columns],
                                                input_df=True,
                                                df_out=True
                                               )



# Combine the numeric and categorical transformations
numeric_categorical_union = FeatureUnion([
                                          ("num_mapper", numeric_imputation_mapper),
                                           ("cat_mapper", categorical_imputation_mapper)
                                         ])



# Custom transformer to convert Pandas DataFrame into Dict (needed for DictVectorizer)
class Dictifier(BaseEstimator, TransformerMixin): 
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = pd.DataFrame(X)
        return X.to_dict('records')



# Create the DictVectorizer object: dv
dv = DictVectorizer(sort=False)

params= {'tree_method':['gpu_hist'], 'predictor':['gpu_predictor']}

# Instantiate the XGBClassifier: xg_cl
xg_cl = xgb.XGBClassifier(objective='binary:logistic',params=params)



# Create full pipeline
pipeline = Pipeline([
                     ("featureunion", numeric_categorical_union),
                       ("dictifier", Dictifier()),
                       ("vectorizer", dv),
                        ("clf", xg_cl)
                    ])

# Create the parameter grid
gbm_param_grid = {
    'clf__learning_rate': np.arange(.05, 1, .05),
    'clf__max_depth': np.arange(3,10, 1),
    'clf__n_estimators': np.arange(50, 200, 50)
}

# Perform RandomizedSearchCV
randomized_pr_auc = RandomizedSearchCV(estimator=pipeline,
                                        param_distributions=gbm_param_grid,
                                        n_iter=2, scoring='average_precision', cv=2, verbose=1)

# Fit the estimator
randomized_pr_auc.fit(X, y)

# Compute metrics
print(randomized_pr_auc.best_score_)
print(randomized_pr_auc.best_estimator_)





# Perform cross-validation
#cross_val_scores = cross_val_score(pipeline, X, y, scoring="average_precision", cv=3)

# Print avg. AUC
#print("3-fold average precision: ", np.mean(cross_val_scores))


# Fit the classifier to the training set
#pipeline.fit(X, y)


# Visualizing feature importances

#xgb.plot_importance(xg_cl)

#plt.savefig('fi')


#import sys
#sys.exit()







