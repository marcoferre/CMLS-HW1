import csv
import numpy as np
import librosa
import os
import json
# import matplotlib.pyplot as plt
# import sklearn.svm
# import IPython.display as ipd
import scipy as sp
from func import *

classes = ["air_conditioner",
           "car_horn",
           "children_playing",
           "dog_bark",
           "drilling",
           "engine_idling",
           "gun_shot",
           "jackhammer",
           "siren",
           "street_music"]

folder_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

tot_features = []

metadata_csv_filename = "D:\\Users\\marco\\Documents\\UrbanSound8K\\UrbanSound8K.csv"
n_mfcc = 13

for folder in folder_ids:

    dict_features = {}

    # debug
    print("testing folder: " + str(folder))
    root = "D:\\Users\\marco\\Documents\\UrbanSound8K\\audio\\fold{}".format(folder)
    for c in classes:

        # debug
        print("\t\ttesting class: " + c)

        class_test_files = extract_info_csv(str(folder), c, metadata_csv_filename)
        n_test_samples = len(class_test_files)

        test_features = np.zeros((n_test_samples, n_mfcc))
        for index, f in enumerate(class_test_files):
            audio, fs = librosa.load(os.path.join(root, f), sr=None)
            mfcc = compute_mfcc(audio, fs, n_mfcc)
            test_features[index, :] = np.mean(mfcc, axis=1)

        print(test_features.tolist(), folder, c)
        dict_features[c] = test_features.tolist()

    tot_features.append(dict_features)

f = open("tot_features.json", "w")
f.write(json.dumps(tot_features))
f.close()

