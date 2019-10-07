import os
import pandas as pd

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
    all_segments = []

    for start in range(0, num_rows - WINDOW_SIZE, STEP):
        window = data_readings[start:start + WINDOW_SIZE]
        all_segments.append(window)

    return all_segments


# Getting labels for phone data
phone_list = []
data_dir = os.path.abspath("./data")
for phone in os.listdir(data_dir):
    phone_list.append(phone)


# Setting up to traverse through CSV files to extract features
for folder in os.listdir(data_dir):
    vibrated_folder = os.path.abspath(os.path.join(data_dir, folder, "vibrated", "50hz"))
    list_of_csv = os.listdir(vibrated_folder)
    for csv in list_of_csv:
        csv_path = os.path.join(vibrated_folder, csv)
        data_segments = create_window(csv_path)
        label = folder











