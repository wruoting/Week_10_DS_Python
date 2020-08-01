from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from sklearn import StandardScaler

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
    
    # Normalize 
    scaler = StandardScaler()
    train_X = scaler.fit_transform(fit_x_training)
    test_X = scaler.transform(fix_x_test)

    
if __name__ == "__main__":
    main()