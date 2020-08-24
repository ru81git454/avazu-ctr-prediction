# Package 
import os
import pandas as pd 
from datetime import datetime
from sklearn.feature_extraction import FeatureHasher
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
from sklearn.externals import joblib 
import matplotlib.pyplot as plt
'''
# Download and unzip file
!kaggle competitions download -c avazu-ctr-prediction
gunzip sampleSubmission, train, test.gz
'''
# parameter
## path 
folder = '/Users/pei/Downloads/avazu-ctr-prediction' # path to folder
train = 'train'                                      # path to training file
test = 'test'                                        # path to testing file
samplesub = 'sampleSubmission'                       # path to sampleSubmission file
submission = 'avazu.csv'                             # path to output file
## data load
chunk_size = 100000                                  # loading data chunk size
train_prop = 0.8                                     # train, valid prop
total_rows = 40428967                                # total row in train
## prepocess

format = '%Y%m%d%H'                                  # time format in dataset
number_hashing_features = 2**25                      # hash trick number of matrics
## define columns
id_col = 'id'                                        # identifier
y_col = 'click'                                      # outcome
datetime_col = 'hour'                                # predictors
categorical_cols = ['C1','banner_pos', 
        'site_id', 'site_domain','site_category', 
        'app_id', 'app_domain', 'app_category', 
        'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type', 
        'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']
## model parameter :SGD
loss = 'log'
learning_rate = 'adaptive'
eta0 = 0.005
penalty = 'elasticnet'

# Function
## convert data type
def to_datetime(df, col = datetime_col, format = format):           #convert str to datetime
    df[col] = df[col].astype(str).apply(lambda x: datetime.strptime('20'+x, format))
    return df

def to_str(df, col):            
    df[col] = df[col].astype(str)
    return df

## Feature  
def add_feature(df, col = datetime_col):                            # add 'hour' feature
    df['hour'] = df[col].apply(lambda x : x.time().hour)
    return df

##hasher trick
import numpy as np
hasher = FeatureHasher(n_features = number_hashing_features,input_type ='string')
preprocessor = Pipeline([('feature_hashing', hasher)])
def preprocess_data(df):
    df = to_datetime(df)
    df = add_feature(df)
    X = df.drop([id_col,y_col],axis =1)
    X = preprocessor.fit_transform(np.asarray(X.astype(str)))
    y = df[y_col]
    return X,y
## Model
# sgd = SGDClassifier(loss= loss,learning_rate=learning_rate,eta0=eta0, penalty=penalty)
# partial_fit_classes = np.array([0, 1])      # to be used in partial_fit method

# Train & Evaluate
## Path 
os.chdir(folder)
# ## View data
# df = pd.read_csv(train,nrows=100000)
# len(df[df['click']==0])/len(df)       # imbalance data
# 100.0*df['click'].sum()/len(df)       # click rate
# df.describe 
# df.dtypes
# df.groupby('click').count()

## Prepare Data
# train_rows = int(train_prop*total_rows)
# valid_rows = total_rows - train_rows
# nr_iterations_train = round(train_rows/chunk_size)
# nr_iterations_valid = round(valid_rows/chunk_size)

# df_train = pd.read_csv( 
#     train, sep=',', chunksize=chunk_size, header=0, nrows=train_rows)
# df_valid = pd.read_csv(
#     train, sep=',', chunksize=chunk_size, skiprows=range(1,train_rows), header=0, nrows=valid_rows)

# ## Training
# print("Training ...")
# count = 0
# for df in df_train:
#     count+=1
#     X_train, y_train = preprocess_data(df)
#     sgd.partial_fit(X_train,y_train, classes=partial_fit_classes)
#     print(count)


## model save
# joblib.dump(sgd, 'sgd.pkl')
## model load
# sgd = joblib.load('sgd.pkl')

## Evaluate
### on train
# print("Log loss on train set ...")
# logloss_sum = 0
# count = 0
# for df in df_train:
#     count += 1
#     X_train, y_train = preprocess_data(df)
#     y_pred = sgd.predict_proba(X_train)
#     log_loss_temp = log_loss(y_train, y_pred)
#     logloss_sum += log_loss_temp
#     if count % 10 == 0:
#         print("Iteration {}/{}, test log loss: {}".format(count,nr_iterations_train,logloss_sum/count))
# print("Final log loss: ", logloss_sum/count)

# ### on valid
# print("Log loss on valid set ...")
# logloss_sum = 0
# count = 0
# for df in df_valid:
#     count += 1
#     X_valid, y_valid = preprocess_data(df)
#     y_pred = sgd.predict_proba(X_valid)
#     log_loss_temp = log_loss(y_valid, y_pred)
#     logloss_sum += log_loss_temp
#     if count % 10 == 0:
#         print("Iteration {}/{}, test log loss: {}".format(count,nr_iterations_valid,logloss_sum/count))
# print("Final log loss: ", logloss_sum/count)


# Deploy
# ## training
# print("Training (train+valid for deploy) ...")
# count = 0
# for df in df_valid:
#     count+=1
#     X_valid, y_valid = preprocess_data(df)
#     sgd.partial_fit(X_valid,y_valid, classes=partial_fit_classes)
#     print(count)

## model save & load
# joblib.dump(sgd, 'sgd_D.pkl')
sgd_D= joblib.load('sgd_D.pkl')

# Data load
df_test = pd.read_csv(test, sep=',', chunksize=chunk_size, header=0, dtype= {'id': 'str'} )  
## Model Fit
print("Fit on test set ...")
logloss_sum = 0
count = 0
with open(submission, 'a') as outfile:
    outfile.write('id,click\n')
    for df in df_test:
        count += 1
        df = to_datetime(df)
        df = add_feature(df)
        id = df[id_col]
        X_test = df.drop([id_col],axis =1)
        X_test = preprocessor.fit_transform(np.asarray(X_test.astype(str)))
        y_pred = sgd_D.predict_proba(X_test)[:,1]
        df_pred = pd.DataFrame(dict(id = id,click = y_pred), columns=['id','click'])
        print(count)
        if count % 10 == 0:
            print("Iteration : {}".format(count))
        df_pred.to_csv(outfile,header=None,index_label=None,index=False)

## output check
output = pd.read_csv(submission)
sample_output = pd.read_csv(samplesub)
print(output.dtypes)
print('outpute shape check:', output.shape == sample_output.shape)

