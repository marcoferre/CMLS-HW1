import csv
import numpy as np
import librosa
import os
import json
# import matplotlib.pyplot as plt
# import sklearn.svm
# import IPython.display as ipd
import scipy as sp

def extract_info_csv(fold, class_name, filename):
    with open(filename, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        out = []
        for row in csv_reader:
            if row["fold"] == fold and row["class"] == class_name:
                out.append(row["slice_file_name"])
    return out


def compute_mfcc(audio, fs, n_mfcc):
    # Compute the spectrogram of the audio signal
    X = np.abs(librosa.stft(
        audio,
        window='hamming',
        n_fft=1024,
        hop_length=512, )
    )

    # Find the weights of the mel filters
    mel = librosa.filters.mel(
        sr=fs,
        n_fft=1024,
        n_mels=40,
        fmin=133.33,
        fmax=6853.8,
    )
    # Apply the filters to spectrogram
    melspectrogram = np.dot(mel, X)
    # Take the logarithm
    log_melspectrogram = np.log10(melspectrogram + 1e-16)

    # Apply the DCT to log melspectrogram to obtain the coefficients
    mfcc = sp.fftpack.dct(log_melspectrogram, axis=0, norm='ortho')[1:n_mfcc + 1]
    return mfcc


def compute_zcr(win, Fs):
    win_sign = np.sign(win)

    N = win.shape[0]

    sign_diff = np.abs(win_sign[:-1] - win_sign[1:])

    zcr = len(sign_diff[sign_diff != 0]) * Fs / N

    return zcr


def compute_speccentr(spec):
    k_axis = np.arange(1, spec.shape[0] + 1)
    centr = np.sum(np.transpose(k_axis) * np.abs(spec)) / np.sum(np.abs(spec))
    return centr


def compute_specdec(spec):
    mul_fact = 1 / np.sum(np.abs(spec[1:]))
    num = np.abs(spec[1:]) - np.tile(np.abs(spec[0]), len(spec) - 1)
    den = np.arange(1, len(spec))
    spectral_decrease = mul_fact * np.sum(num / den)
    return spectral_decrease


def compute_cm_multiclass(gt, predicted):
    classes = np.unique(gt)
    CM = np.zeros((len(classes), len(classes)))
    for i in np.arange(len(classes)):
        pred_class = predicted[gt == i]

        for j in np.arange(len(pred_class)):
            CM[i, int(pred_class[j])] = CM[i, int(pred_class[j])] + 1
    return CM


def compute_metrics(gt_labels, predicted_labels):

    TP = np.sum(np.logical_and(predicted_labels == 1, gt_labels == 1))
    FP = np.sum(np.logical_and(predicted_labels == 1, gt_labels == 0))
    TN = np.sum(np.logical_and(predicted_labels == 0, gt_labels == 0))
    FN = np.sum(np.logical_and(predicted_labels == 0, gt_labels == 1))
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1_score = 2 * precision * recall / (precision + recall)
    print("Results : \n accuracy = {} \n precision = {} \n recall = {} \n F1 score = {}".format(
        accuracy, precision, recall, F1_score))
