import os
import pandas as pd
import features_extraction
import numpy as np
import matplotlib.pyplot as plt

# Constant to declare for the parameters for the sliding window
WINDOW_SIZE = 8
STEP = 2


# Data path should arrive in the format of <phone>/<vibrated>/<frequency>/<.csv> file
def create_window(data_path):
    columns = ["x", "y", "z"]
    df = pd.read_csv(data_path, usecols=[1, 2, 3], header=None)
    # Assigning columns to the data set
    df.columns = columns

    data_readings = df.values
    num_rows = len(data_readings)
    all_window = []

    for start in range(0, num_rows - WINDOW_SIZE, STEP):
        window = data_readings[start:start + WINDOW_SIZE]
        all_window.append(window)

    return all_window


def create_segments(data_path):
    columns = ["x", "y", "z"]
    df = pd.read_csv(data_path, usecols=[1, 2, 3], header=None)
    # Assigning columns to the data set
    df.columns = columns
    data_readings = df.values
    num_rows = len(data_readings)
    all_segments = []

    for start in range(0, num_rows - WINDOW_SIZE, WINDOW_SIZE):
        segment = data_readings[start:start+WINDOW_SIZE]
        all_segments.append(segment)

    return all_segments


# Getting labels for phone data
phone_list = []
data_dir = os.path.abspath("./data")
for phone in os.listdir(data_dir):
    phone_list.append(phone)

# Setting up to traverse through CSV files to extract features
data_segments = []
for folder in os.listdir(data_dir):
    vibrated_folder = os.path.abspath(os.path.join(data_dir, folder))
    list_of_csv = os.listdir(vibrated_folder)
    for csv in list_of_csv:
        csv_path = os.path.join(vibrated_folder, csv)
        data_window = create_window(csv_path)
        data_segments = create_segments(csv_path)


# # data = "D:\\Y4S1\\CS4276\\device-fingerprint\\data\\blue_huawei\\gyro_100hz_14102019_204127.csv"
# data = "D:\\Y4S1\\CS4276\\device-fingerprint\\data\\htc_u11\\gyro_100hz_16102019_205643.csv"
# # data = "D:\\Y4S1\\CS4276\\device-fingerprint\\data\\mi_max\\gyro_100hz_14102019_212145.csv"
# data1 = "D:\\Y4S1\\CS4276\\device-fingerprint\\data\\htc_u11\\gyro_100hz_16102019_205729.csv"
#
# name1 ="htcu11"
# name2 ="htcu11"
#
#
# # For Black huawei
# data_segments = create_segments(data)
# data_windows = create_window(data)
#
# # for blue huawei
# data1_segments = create_segments(data1)
# data1_windows = create_window(data1)
#
# # for black huawei
# features_segment = []
# for i in data_segments:
#     features_segment.append(features_extraction.extract_features(i))
#
# # for blue huawei
# features_segment1 = []
# for i in data1_segments:
#     features_segment1.append(features_extraction.extract_features(i))
#
#
# for j in range(len(features_segment[0])):
#     list1 = []
#     list2 = []
#     for i in features_segment:
#         list1.append(i[j])
#
#     for i in features_segment1:
#         list2.append(i[j])
#
#     plt.plot(list1, label=name1)
#     plt.plot(list2, label=name2)
#     plt.xlabel("segments")
#     plt.ylabel("readings")
#     legend = plt.legend(loc='upper center', shadow=True, fontsize='large')
#     if not os.path.exists("./" + name1 + "_" + name2):
#         os.mkdir("./" + name1 + "_" + name2)
#     plt.savefig("./" + name1 + "_" + name2 + "/" + str(j) + name1 + "_" + name2 + '.png')
#     plt.clf()

# features_window = []
# features_segment = []
# for i in data_window:
#     features_window.extend(features_extraction.extract_features(i))
#
# for i in data_segments:
#     features_segment.extend(features_extraction.extract_features(i))
#
# # print(features_window)
# print(len(features_segment))












