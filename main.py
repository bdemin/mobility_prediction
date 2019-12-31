import pandas as pd

# Read data
train_df = pd.read_csv('mobility_fake_dataset.csv')
#check data has been read in properly
train_df.head()

#create a dataframe with all training data except the target column
train_X = train_df.drop(columns=['mobility'])

#check that the target variable has been removed
train_X.head()

#create a dataframe with only the target column
train_y = train_df[['mobility']]

#view dataframe
train_y.head()

