import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# pre-processing
from scipy import signal
from scipy.fftpack import fft
from scipy.signal import blackman
from scipy.signal import welch
from scipy import signal
from sklearn import preprocessing

# feature extraction
import features_extraction
from detect_peaks import detect_peaks

# classification model
from sklearn import tree
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

import sklearn.linear_model as sk

warning = False
detail = True

# set warning
if not warning:
    warnings.filterwarnings('ignore')

# Constant to declare for the parameters for the sliding window
WINDOW_SIZE = 1024
STEP = 512
order = 3
f_sample = 100
f_cutoff = 30
nyquist_rate = f_cutoff/(f_sample/2)

N = WINDOW_SIZE # samples
f_s = 100 # frequency
t_n = N/f_s # sec
T = t_n / N

sample_rate = 1

denominator = 10 # mph for feature extraction

# label phone models
phone_models = {
    "black_huawei": 1,
    "galaxy_note5": 2,
    "htc_u11": 3,
    "pixel_2": 4,
    "blue_huawei": 5,
    "galaxy_tabs": 6,
    "mi_max": 7
}

# Data path should arrive in the format of <phone>/<vibrated>/<frequency>/<.csv> file
def create_window(data_path):
    columns = ["x", "y", "z"]
    df = pd.read_csv(data_path, usecols=[1, 2, 3], header=None)
    # Assigning columns to the data set
    df.columns = columns

    data_readings = df.values

    # noise reduction
    
    for axis in range(0,3):
        # order 3 lowpass butterworth filter
        b, a = signal.butter(order, nyquist_rate)
        data_readings[:,axis] = signal.filtfilt(b, a, data_readings[:,axis])
    
    # normalization
    min_max_scaler = preprocessing.MinMaxScaler()   
    data_readings = min_max_scaler.fit_transform(data_readings)


    num_rows = len(data_readings)
    all_window = []

    for start in range(0, num_rows - WINDOW_SIZE, STEP):
        window = data_readings[start:start + WINDOW_SIZE]
        all_window.append(window)

    return all_window

# fft
def get_fft_values(y_values, T, N, f_s):
    f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fft_values_ = fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return f_values, fft_values

# psd
def get_psd_values(y_values, T, N, f_s):
    f_values, psd_values = welch(y_values, fs=f_s)
    return f_values, psd_values

# auto-correlation
def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[len(result)//2:]

def get_autocorr_values(y_values, T, N, f_s):
    autocorr_values = autocorr(y_values)
    x_values = np.array([T * jj for jj in range(0, N)])
    return x_values, autocorr_values


def get_values(y_values, T, N, f_s):
    y_values = y_values
    x_values = [sample_rate * kk for kk in range(0,len(y_values))]
    return x_values, y_values

def read_signals(filename):
    with open(filename, 'r') as fp:
        data = fp.read().splitlines()
        data = map(lambda x: x.rstrip().lstrip().split(), data)
        data = [list(map(float, line)) for line in data]
        data = np.array(data, dtype=np.float32)
    return data

def read_labels(filename):        
    with open(filename, 'r') as fp:
        activities = fp.read().splitlines()
        activities = list(map(int, activities))
    return np.array(activities)

def get_first_n_peaks(x,y,no_peaks=5):
    x_, y_ = list(x), list(y)
    if len(x_) >= no_peaks:
        return x_[:no_peaks], y_[:no_peaks]
    else:
        missing_no_peaks = no_peaks-len(x_)
        return x_ + [0]*missing_no_peaks, y_ + [0]*missing_no_peaks
    
def get_features(x_values, y_values, mph):
    indices_peaks = detect_peaks(y_values, mph=mph)
    peaks_x, peaks_y = get_first_n_peaks(x_values[indices_peaks], y_values[indices_peaks])
    return peaks_x + peaks_y

def extract_features_labels(dataset, labels, T, N, f_s, denominator):
    percentile = 5
    list_of_features = []
    list_of_labels = []
    for signal_no in range(0, len(dataset)):
        features = []
        list_of_labels.append(labels[signal_no])
        for signal_comp in range(0,dataset.shape[2]):
            signal = dataset[signal_no, :, signal_comp]
            
            signal_min = np.nanpercentile(signal, percentile)
            signal_max = np.nanpercentile(signal, 100-percentile)
            #ijk = (100 - 2*percentile)/10
            mph = signal_min + (signal_max - signal_min)/denominator
            
            features += get_features(*get_psd_values(signal, T, N, f_s), mph)
            features += get_features(*get_fft_values(signal, T, N, f_s), mph)
            features += get_features(*get_autocorr_values(signal, T, N, f_s), mph)
        list_of_features.append(features)
    return np.array(list_of_features), np.array(list_of_labels)

def retrieve(data_dir):
    # Getting labels for phone data
    phone_list = []
    labels = []
    for phone in os.listdir(data_dir):
        #print(phone)
        phone_list.append(phone)

    # Setting up to traverse through CSV files to extract features
    data_window = []
    data_windows = []
    for folder in os.listdir(data_dir):
        vibrated_folder = os.path.abspath(os.path.join(data_dir, folder))
        list_of_csv = os.listdir(vibrated_folder)

        for csv in list_of_csv:
            csv_path = os.path.join(vibrated_folder, csv)
            data_window = create_window(csv_path)

			# add labels
            labels = labels + [phone_models[folder]]*np.shape(data_window)[0]
            data_windows.extend(data_window)

    return data_windows, labels

def evaluate(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    print("\n" + "-"*15 + " " + type(model).__name__ + " " + "-"*15)
    print("Accuracy on training set is : {}".format(model.score(x_train, y_train)))
    print("Accuracy on test set is : {}".format(model.score(x_test, y_test)))

    if detail:
        y_test_pred = model.predict(x_test)
        print(classification_report(y_test, y_test_pred))

train_signals, train_labels = retrieve(os.path.abspath("./100hz_data"))
test_signals, test_labels = retrieve(os.path.abspath("./test"))

# Plot to see window details
labels = ['x-component', 'y-component', 'z-component']
colors = ['r', 'g', 'b']
suptitle = "Different signals for the activity: {}"

xlabels = ['Time [sec]', 'Freq [Hz]', 'Freq [Hz]', 'Time lag [s]']
ylabel = 'Amplitude'
axtitles = [['Gyro'],
            ['FFT gyro'],
            ['PSD gyro'],
            ['Autocorr gyro']
           ]
axtitles = [['Gyro'],
            ['FFT gyro'],
            ['PSD gyro'],
            ['Autocorr gyro']
           ]

list_functions = [get_values, get_fft_values, get_psd_values, get_autocorr_values]

signal_no = 0
signals = train_signals[signal_no][:][:]
label = train_labels[signal_no]
activity_name = list(phone_models.keys())[list(phone_models.values()).index(label)]

f, axarr = plt.subplots(nrows=4, ncols=1, figsize=(12,12))
f.suptitle(suptitle.format(activity_name), fontsize=16)

for row_no in range(0,4):
    for comp_no in range(0,3):
        col_no = comp_no // 3
        plot_no = comp_no % 3
        color = colors[plot_no]
        label = labels[plot_no]

        axtitle  = axtitles[row_no][col_no]
        xlabel = xlabels[row_no]
        value_retriever = list_functions[row_no]

        ax = axarr[row_no]
        ax.set_title(axtitle, fontsize=16)
        ax.set_xlabel(xlabel, fontsize=16)
        if col_no == 0:
            ax.set_ylabel(ylabel, fontsize=16)

        signal_component = signals[:, comp_no]
        x_values, y_values = value_retriever(signal_component, T, N, f_s)
        ax.plot(x_values, y_values, linestyle='-', color=color, label=label)
        if row_no > 0:
            max_peak_height = 0.1 * np.nanmax(y_values)
            indices_peaks = detect_peaks(y_values, mph=max_peak_height)
            ax.scatter(x_values[indices_peaks], y_values[indices_peaks], c=color, marker='*', s=60)
        if col_no == 2:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))            
plt.tight_layout()
plt.subplots_adjust(top=0.90, hspace=0.6)
plt.show()
# End plot

### Start Train & Test
# convert to np array
train_signals = np.asarray(train_signals)
test_signals = np.asarray(test_signals)

# prepare train and test sets
x_train, y_train = extract_features_labels(train_signals, train_labels, T, N, f_s, denominator)
x_test, y_test = extract_features_labels(test_signals, test_labels, T, N, f_s, denominator)

print("\n" + "*"*20 + " Classification Model " + "*"*20)

# RandomForest
model = RandomForestClassifier(n_estimators=1000)
evaluate(model, x_train, x_test, y_train, y_test)

# MaxVoting
model1 = KNeighborsClassifier(n_neighbors=3)
model2 = tree.DecisionTreeClassifier(random_state=1)
model3 = sk.LogisticRegression(random_state=1, solver='liblinear')
model = VotingClassifier(estimators=[('kn', model1), ('dt', model2), ('lr', model3)], voting='hard')
evaluate(model, x_train, x_test, y_train, y_test)

# XGB Classifier
model=xgb.XGBClassifier(random_state=1,learning_rate=0.01)
evaluate(model, x_train, x_test, y_train, y_test)

# Baggging Classifier
model = BaggingClassifier(tree.DecisionTreeClassifier(random_state=1))
evaluate(model, x_train, x_test, y_train, y_test)






