from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn import svm

def transform_trading_days_to_trading_weeks(df):
    '''
    df: dataframe of relevant data
    returns: dataframe with processed data, only keeping weeks, their open and close for said week
    '''
    trading_list = deque()
    # Iterate through each trading week
    for trading_week, df_trading_week in df.groupby(['Year','Week_Number']):
        classification =  df_trading_week.iloc[0][['Classification']].values[0]
        opening_day_of_week = df_trading_week.iloc[0][['Open']].values[0]
        closing_day_of_week = df_trading_week.iloc[-1][['Close']].values[0]
        trading_list.append([trading_week[0], trading_week[1], opening_day_of_week, closing_day_of_week, classification])
    trading_list_df = pd.DataFrame(np.array(trading_list), columns=['Year', 'Trading Week', 'Week Open', 'Week Close', 'Classification'])
    return trading_list_df

def make_trade(cash, open, close):
    '''
    cash: float of cash on hand
    open: float of open price
    close: float of close price
    returns: The cash made from a long position from open to close
    '''
    shares = np.divide(cash, open)
    return np.multiply(shares, close)

def trading_strategy(trading_df, prediction_label, weekly_balance=100):
    '''
    trading_df: dataframe of relevant weekly data
    prediction_label: the label for which we're going to trade off of
    returns: A df of trades made based on Predicted Labels
    '''
    # The weekly balance we will be using
    weekly_balance_acc = weekly_balance
    trading_history = deque()
    index = 0
    
    while(index < len(trading_df.index) - 1):
        trading_week_index = index
        if weekly_balance_acc != 0:
            # Find the next consecutive green set of weeks and trade on them
            while(trading_week_index < len(trading_df.index) - 1 and trading_df.iloc[trading_week_index][[prediction_label]].values[0] == 'GREEN'):
                trading_week_index += 1
            green_weeks = trading_df.iloc[index:trading_week_index][['Week Open', 'Week Close']]
            # Check if there are green weeks, and if there are not, we add a row for trading history
            if len(green_weeks.index) > 0:
                # Buy shares at open and sell shares at close of week
                green_weeks_open = float(green_weeks.iloc[0][['Week Open']].values[0])
                green_weeks_close = float(green_weeks.iloc[-1][['Week Close']].values[0])
                # We append the money after we make the trade
                weekly_balance_acc = make_trade(weekly_balance_acc, green_weeks_open, green_weeks_close)
            # Regardless of whether we made a trade or not, we append the weekly cash and week over
            trading_history.append([trading_df.iloc[trading_week_index][['Year']].values[0],
                trading_df.iloc[trading_week_index][['Trading Week']].values[0],
                weekly_balance_acc])
        else:
            # If we have no money we will not be able to trade
            trading_history.append([trading_df.iloc[trading_week_index][['Year']].values[0],
                    trading_df.iloc[trading_week_index][['Trading Week']].values[0],
                    weekly_balance_acc])
        index = trading_week_index+1
    trading_hist_df = pd.DataFrame(np.array(trading_history), columns=['Year', 'Trading Week', 'Balance'])
    trading_hist_df['Balance'] = np.round(trading_hist_df[['Balance']].astype(float), 2)

    return trading_hist_df



def main():
    ticker='WMT'
    file_name_self_labels = 'WMT_Labeled_Weeks_Self.csv'
    file_name = 'WMT_weekly_return_volatility.csv'
    df = pd.read_csv(file_name, encoding='ISO-8859-1')
    df_2018 = df[df['Year'] == 2018]
    df_2019 = df[df['Year'] == 2019]


    fit_x_training = df_2018[['mean_return', 'volatility']]
    fit_y_training = df_2018[['Classification']].values
    fix_x_test = df_2019[['mean_return', 'volatility']]
    fit_y_test = df_2019[['Classification']].values
    
    # Normalize test data with training fit
    normalizer = Normalizer()
    normalizer.fit(fit_x_training)
    train_X = normalizer.transform(fit_x_training)
    test_X = normalizer.transform(fix_x_test)

    # Implement a linear kernel for SVM
    svm_classifier = svm.SVC(kernel='linear')
    svm_classifier.fit(train_X, fit_y_training.ravel())

    predict_y = svm_classifier.predict(test_X)
    # This is the same as the score calculation native to the function
    accuracy = np.round(np.multiply(np.mean(predict_y == fit_y_test), 100), 2)

    print('Question 1:')
    print('Linear SVM post normalization of year 2 with year 1 fit.')
    print("Accuracy for year 2 is: {}%".format(accuracy))

    print('\nQuestion 2:')
    print('Decision Tree Confusion matrix')
    confusion_matrix_array_decision_tree = confusion_matrix(fit_y_test, predict_y)
    confusion_matrix_df_decision_tree = pd.DataFrame(confusion_matrix_array_decision_tree, columns= ['Predicted: GREEN', 'Predicted: RED'], index=['Actual: GREEN', 'Actual: RED'])
    print(confusion_matrix_df_decision_tree)

    print('\nQuestion 3:')
    total_data_points = len(fit_y_test)
    true_positive_number = confusion_matrix_df_decision_tree['Predicted: GREEN']['Actual: GREEN']
    true_positive_rate = np.round(np.multiply(np.divide(true_positive_number, total_data_points), 100), 2)
    true_negative_number = confusion_matrix_df_decision_tree['Predicted: RED']['Actual: RED']
    true_negative_rate = np.round(np.multiply(np.divide(true_negative_number, total_data_points), 100), 2)
    print('True positive rate: {}%'.format(true_positive_rate))
    print('True negative rate: {}%'.format(true_negative_rate))

    print('\nQuestion 4:')
    svm_classifier = svm.SVC(kernel='rbf', gamma='scale')
    svm_classifier.fit(train_X, fit_y_training.ravel())
    predict_y = svm_classifier.predict(test_X)
    accuracy = np.round(np.multiply(np.mean(predict_y == fit_y_test), 100), 2)
    print("Accuracy for year 2 is: {}%".format(accuracy))
    print('It seems like gaussian and linear SVM both do a poor job of separation of the test data. They both have consistent ')
    print('accuracy of determining every point as RED.')

    print('\nQuestion 5:')
    svm_classifier = svm.SVC(kernel='poly', degree=2, gamma='scale')
    svm_classifier.fit(train_X, fit_y_training.ravel())
    predict_y = svm_classifier.predict(test_X)
    accuracy = np.round(np.multiply(np.mean(predict_y == fit_y_test), 100), 2)
    print("Accuracy for year 2 is: {}%".format(accuracy))
    print('It seems like polynomial, gaussian, and linear SVM all do a poor job of separation of the test data. They all have consistent ')
    print('accuracy of determining every point as RED.')

    print('\nQuestion 6:')
    # Implement a linear kernel for SVM
    svm_classifier = svm.SVC(kernel='linear')
    svm_classifier.fit(train_X, fit_y_training.ravel())
    predict_y = svm_classifier.predict(test_X)

    df = pd.read_csv(file_name_self_labels, encoding='ISO-8859-1')
    df_trading_weeks = transform_trading_days_to_trading_weeks(df)
    trading_weeks_2019 = df_trading_weeks[df_trading_weeks['Year'] == '2019']
    buy_and_hold = np.full(len(trading_weeks_2019.index), 'GREEN')

    trading_weeks_2019.insert(len(trading_weeks_2019.columns), "Buy and Hold", buy_and_hold, allow_duplicates=True)
    trading_weeks_2019.insert(len(trading_weeks_2019.columns), "Predicted Labels SVM Linear", predict_y, allow_duplicates=True)
    balance_end_svm_linear = trading_strategy(trading_weeks_2019, "Predicted Labels SVM Linear")[['Balance']].values[-1][0]
    balance_end_hold = trading_strategy(trading_weeks_2019, "Buy and Hold")[['Balance']].values[-1][0]

    print('With buy and hold: ${}'.format(balance_end_hold))
    print('With linear SVM: ${}'.format(balance_end_svm_linear))

    print('Our SVM has classified all the data to be red, so no trading was done for year 2. Therefore, buy and hold was a better strategy.')

if __name__ == "__main__":
    main()