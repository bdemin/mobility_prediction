import pandas as pd

    # Read data
    train_df = pd.read_csv('mobility_fake_dataset.csv')

    #create a dataframe with all training data except the target column
    train_X = train_df.drop(columns=['mobility'])


    #create a dataframe with only the target column
    train_y = train_df[['mobility']]



from keras.models import Sequential
from keras.layers import Dense
#create model
model = Sequential()

#get number of columns in training data
n_cols = train_X.shape[1]

#add model layers
model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

#compile model using mse as a measure of model performance
model.compile(optimizer='adam', loss='mean_squared_error')

from keras.callbacks import EarlyStopping
#set early stopping monitor so the model stops training when it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=3)
#train model
model.fit(train_X, train_y, validation_split=0.2, epochs=30, callbacks=[early_stopping_monitor])

#example on how to use our newly trained model on how to make predictions on unseen data (we will pretend our new data is saved in a dataframe called 'test_X').
# test_y_predictions = model.predict(test_X)

# #training a new model on the same data to show the effect of increasing model capacity

# #create model
# model_mc = Sequential()

# #add model layers
# model_mc.add(Dense(200, activation='relu', input_shape=(n_cols,)))
# model_mc.add(Dense(200, activation='relu'))
# model_mc.add(Dense(200, activation='relu'))
# model_mc.add(Dense(1))

# #compile model using mse as a measure of model performance
# model_mc.compile(optimizer='adam', loss='mean_squared_error')
# #train model
# model_mc.fit(train_X, train_y, validation_split=0.2, epochs=30, callbacks=[early_stopping_monitor])

