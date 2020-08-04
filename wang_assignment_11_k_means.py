from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn import svm
import matplotlib.pyplot as plt

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

    distortion_list = deque()
    for k in range(1, 9):
        kmeans = KMeans(n_clusters=k, init='random').fit(train_X)
        predict_y = kmeans.predict(test_X)
        distortion = np.round(kmeans.inertia_)
        distortion_list.append(distortion)
    distortion_list = np.array(distortion_list)

    plt.plot(np.arange(1,9), distortion_list)
    plt.title('K means accuracy vs clusters')
    plt.ylabel('Distortion')
    plt.xlabel('Number of cluster')
    plt.show()

    print('\nQuestion 1:')
    print('The number of clusters optimal from the knee method is 5')
    print('\nQuestion 2:')
    k = 5
    kmeans = KMeans(n_clusters=k, init='random').fit(train_X)
    # List of classifications with their corresponding clusters
    predict_y = kmeans.predict(test_X)
    classification_y = fit_y_test.T[0]
    green = {}
    red = {}
    for cluster, classification in zip(predict_y, classification_y):
        if classification == 'GREEN':
            if green.get(cluster) is not None:
                green[cluster]+=1
            else:
                green[cluster] = 1
        elif classification == 'RED':
            if red.get(cluster) is not None:
                red[cluster]+=1
            else:
                red[cluster] = 1
    # Calculate per cluster percentages
    total_array = deque()
    # Creating a list of clusters with more than 90%
    purity_arr = []
    for green, red in zip(sorted(green.items()), sorted(red.items())):
        total = green[1] + red[1]
        percent_green = np.divide(green[1], total)
        percent_red = np.divide(red[1], total)
        total_array.append([percent_green, percent_red])
        # If either are greater than 90%, we append to purity
        if percent_green >= 0.9 or percent_red >= 0.9:
            purity_arr.append(len(total_array)-1)
    total_array = np.array(total_array)

    for index, value in enumerate(total_array):
        print('Cluster {} is comprised of {}% green and {}% red'.format(index+1, np.round(np.multiply(value[0], 100), 2),  np.round(np.multiply(value[1], 100), 2)))
    print('\nQuestion 3:')
    print('Using 90% as a cutoff for purity.')
    if len(purity_arr) > 0:
        for value in purity_arr:
            green = np.round(np.multiply(total_array[value][0], 100), 2)
            red = np.round(np.multiply(total_array[value][1], 100), 2)
            print('Cluster {} is pure with  {}% green and {}% red'.format(value+1, green, red))
    else:
        print('No pure clusters were found.')
    

if __name__ == "__main__":
    main()