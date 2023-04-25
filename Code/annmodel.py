import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from datetime import datetime

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

#Data preprocessing: import dataset
company1_train = pd.read_csv('/content/GS_Training.csv')
company1_test = pd.read_csv('/content/GS_Testing.csv')
company2_train = pd.read_csv('/content/JNJ_Training.csv')
company2_test = pd.read_csv('/content/JNJ_Testing.csv')
company3_train = pd.read_csv('/content/JPM_Training.csv')
company3_test = pd.read_csv('/content/JPM_Testing.csv')
company4_train = pd.read_csv('/content/NKE_Training.csv')
company4_test = pd.read_csv('/content/NKE_Testing.csv')
company5_train = pd.read_csv('/content/PFE_Training.csv')
company5_test = pd.read_csv('/content/PFE_Testing.csv')

#Function for calculating FIRST parameter (each day's high - low)
def calculate_high_low_diff(df):
    high_low_values = df["High"] - df["Low"]
    new_df = pd.DataFrame({"Date": df["Date"], "High-Low": high_low_values})
    return new_df

goldman_high_minus_low = calculate_high_low_diff(company1_train)
johnson_high_minus_low = calculate_high_low_diff(company2_train)
jpmorgan_high_minus_low = calculate_high_low_diff(company3_train)
nike_high_minus_low = calculate_high_low_diff(company4_train)
pfizer_high_minus_low = calculate_high_low_diff(company5_train)
print(goldman_high_minus_low)

#Function for calculating SECOND parameter (each day's close - open)
def calculate_close_open_diff(df):
    new_df = pd.DataFrame()
    new_df['Date'] = df['Date']
    new_df['Close-Open'] = df['Close'] - df['Open']
    return new_df

#Variable 2
#New DataFrame for each company's Close-Open Variable
goldman_close_minus_open = calculate_close_open_diff(company1_train)
johnson_close_minus_open = calculate_close_open_diff(company2_train)
jpmorgan_close_minus_open = calculate_close_open_diff(company3_train)
nike_close_minus_open = calculate_close_open_diff(company4_train)
pfizer_close_minus_open = calculate_close_open_diff(company5_train)
print(goldman_close_minus_open)

#Function for calculating 3rd|4th|5th parameters (7|14|21 Day Moving Average) and gets rid of NaN values
def moving_average(dataframe, days):
    #Create a new dataframe with the moving average
    ma_df = pd.DataFrame()
    #ma_df['Date'] = dataframe['Date']
    ma_df['Moving Avg'] = dataframe['Close'].rolling(window=days).mean()

    #Add a column with the date of each week
    ma_df['Date'] = dataframe['Date']

    #Remove NaN values
    ma_df = ma_df.dropna()

    return ma_df

#Variable 3
#New DataFrame for each company's 7 day Moving Average
goldman_7D_movingAverage = moving_average(company1_train, 7)
johnson_7D_movingAverage = moving_average(company2_train, 7)
jpmorgan_7D_movingAverage = moving_average(company3_train, 7)
nike_7D_movingAverage = moving_average(company4_train, 7)
pfizer_7D_movingAverage = moving_average(company5_train, 7)
print(goldman_7D_movingAverage)

#Variable 4
#New DataFrame for each company's 14 day Moving Average
goldman_14D_movingAverage = moving_average(company1_train, 14)
johnson_14D_movingAverage = moving_average(company2_train, 14)
jpmorgan_14D_movingAverage = moving_average(company3_train, 14)
nike_14D_movingAverage = moving_average(company4_train, 14)
pfizer_14D_movingAverage = moving_average(company5_train, 14)
print(goldman_14D_movingAverage)

#Variable 5
#New DataFrame for each company's 21 day Moving Average
goldman_21D_movingAverage = moving_average(company1_train, 21)
johnson_21D_movingAverage = moving_average(company2_train, 21)
jpmorgan_21D_movingAverage = moving_average(company3_train, 21)
nike_21D_movingAverage = moving_average(company4_train, 21)
pfizer_21D_movingAverage = moving_average(company5_train, 21)
print(goldman_21D_movingAverage)

#Function for calculating 6th parameter: Std for past 7 days and gets rid of NaN values
def calculate_7day_std_dev(df):
    #Create a new DataFrame with just the date and close columns
    close_df = df[['Date', 'Close']].copy()

    #Calculate the 7-day rolling standard deviation of the closing price
    close_df['7 Day Std Dev'] = close_df['Close'].rolling(window=7).std()

    #Drop the first 6 rows (since we don't have enough data to calculate the 7-day std dev for them)
    close_df = close_df.dropna()

    return close_df

#Variable 6
#New DataFrame for each company's Past 7 day std
goldman_7D_STD = calculate_7day_std_dev(company1_train)
johnson_7D_STD = calculate_7day_std_dev(company2_train)
jpmorgan_7D_STD = calculate_7day_std_dev(company3_train)
nike_7D_STD = calculate_7day_std_dev(company4_train)
pfizer_7D_STD = calculate_7day_std_dev(company5_train)
print(goldman_7D_STD)

def merge_dataframes(df1, df2, df3, df4, df5, df6, on_column):
    #Merge the first two dataframes
    merged_df = pd.merge(df1, df2, on=on_column)

    #Merge the remaining dataframes
    merged_df = pd.merge(merged_df, df3, on=on_column)
    merged_df = pd.merge(merged_df, df4, on=on_column)
    merged_df = pd.merge(merged_df, df5, on=on_column)
    merged_df = pd.merge(merged_df, df6, on=on_column)

    return merged_df

company1_InputVars = merge_dataframes(goldman_high_minus_low, goldman_close_minus_open, goldman_7D_movingAverage, goldman_14D_movingAverage, goldman_21D_movingAverage, goldman_7D_STD, 'Date')
company2_InputVars = merge_dataframes(johnson_high_minus_low, johnson_close_minus_open, johnson_7D_movingAverage, johnson_14D_movingAverage, johnson_21D_movingAverage, johnson_7D_STD, 'Date')
company3_InputVars = merge_dataframes(jpmorgan_high_minus_low, jpmorgan_close_minus_open, jpmorgan_7D_movingAverage, jpmorgan_14D_movingAverage, jpmorgan_21D_movingAverage, jpmorgan_7D_STD, 'Date')
company4_InputVars = merge_dataframes(nike_high_minus_low, nike_close_minus_open, nike_7D_movingAverage, nike_14D_movingAverage, nike_21D_movingAverage, nike_7D_STD, 'Date')
company5_InputVars = merge_dataframes(pfizer_high_minus_low, pfizer_close_minus_open, pfizer_7D_movingAverage, pfizer_14D_movingAverage, pfizer_21D_movingAverage, pfizer_7D_STD, 'Date')
print(company5_InputVars)

#Set a random seed for reproducibility
np.random.seed(77)

companies = ['Goldman Sachs', 'Johnson & Johnson', 'JP Morgan and Co.', 'Nike', 'Pfizer Inc.']
scaler = StandardScaler()

#Creating the ANN Model
#Creating sequential object to define the layers
classifier = Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu',
                     input_dim=5))

#Adding the second hidden layer
classifier.add(Dense(units=4, kernel_initializer='uniform', activation='relu'))

#Adding the third hidden layer
classifier.add(Dense(units=2, kernel_initializer='uniform', activation='relu'))

#Adding the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='linear'))

#Compiling the ANN
classifier.compile(optimizer=Adam(lr=0.001), loss='mse', metrics=['mae'])

#Iterate over each company's data and train and test model
for i in range(1, 6):
    #Set a random seed for reproducibility
    np.random.seed(77)
    #Selecting training and testing data for current company
    X_train = globals()[f'company{i}_train'].iloc[:, 2:].values
    y_train = globals()[f'company{i}_train'].iloc[:, 1].values
    X_test = globals()[f'company{i}_test'].iloc[:, 2:].values
    y_test = globals()[f'company{i}_test'].iloc[:, 1].values

    #Scaling the input data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #Fitting the ANN to the training set
    classifier.fit(X_train_scaled, y_train, batch_size=32, epochs=100, verbose=0)

    #Predicting the test set results
    y_pred = classifier.predict(X_test_scaled)
    y_pred = y_pred.reshape(-1) #reshape to 1D array

    #Calculating the evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    mbe = np.mean(y_pred - y_test)

    #Printing the values for each company in the format: Company: RMSE MAPE MBE
    print(f"{companies[i-1]}: {rmse:.2f} {mape:.2f}% {mbe:.4f}")

    #Create datetime objects for x-axis labels
    dates = globals()[f'company{i}_test'].iloc[:, 0].values
    dates = [datetime.strptime(date, '%Y-%m-%d') for date in dates]

    #Plot actual stock prices and predicted stock prices for current company
    plt.plot(dates, y_test, label='Actual Stock Price')
    plt.plot(dates, y_pred, label='Predicted Stock Price')
    plt.title(f'Stock Price Prediction for Company {i}')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.xticks(rotation=45, fontsize=8)
    plt.legend()
    plt.show()

#Plotting tool to visualize ANN Model
plot_model(classifier, to_file='model_plot.png', show_shapes=True, show_layer_names=True)