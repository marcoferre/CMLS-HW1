import json
import sklearn.svm
import numpy
from func import *

classes = ['air_conditioner',
           'car_horn',
           'children_playing',
           'dog_bark',
           'drilling',
           'engine_idling',
           'gun_shot',
           'jackhammer',
           'siren',
           'street_music']

cm_multiclasses_sum = np.zeros((len(classes), len(classes)))

tot_train_features = {}

f = open("output/tot_features.json", "r")
f_str = f.read()

tot_features = json.loads(f_str)

SVM_parameters = {
    'C': 1,
    'kernel': 'rbf',
}

folder_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# compute the minimum nr of elements for all classes
min_nr = 100
for e in folder_ids:
    for c in classes:
        if len(tot_features[e][c]) < min_nr:
            min_nr = len(tot_features[e][c])


#list all the train features
for f_test_id in folder_ids:
    sub_folder_id = [x for x in folder_ids if x != f_test_id]

    dict_train_features = {}

    for f_train_id in sub_folder_id:
        for c in classes:
            if c not in dict_train_features:
                dict_train_features[c] = []

            tmp = tot_features[f_train_id][c][:min_nr]
            dict_train_features[c].extend(tmp)

    tot_train_features[f_test_id] = dict_train_features


#compute tests
folder_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
for f_test_id in folder_ids:
    i = 0

    x_train = {}
    y_train = {}
    x_test = {}
    y_test = {}
    x_train_normalized = {}
    x_test_normalized = {}

    for c in classes:
        x_train[c] = tot_train_features[f_test_id][c]

        y_train[c] = np.ones(len(x_train[c]), ) * i

        x_test[c] = tot_features[f_test_id][c]

        y_test[c] = np.ones(len(x_test[c]), ) * i

        i += 1

    y_test_mc = np.concatenate(get_tupla(y_test), axis=0)

    feat_max = np.max(np.concatenate(get_tupla(x_train), axis=0))
    feat_min = np.min(np.concatenate(get_tupla(x_train), axis=0))

    for c in classes:
        x_train_normalized[c] = (x_train[c] - feat_min) / (feat_max - feat_min)
        x_test_normalized[c] = (x_test[c] - feat_min) / (feat_max - feat_min)


    x_test_mc_normalized = np.concatenate(get_tupla(x_test_normalized), axis=0)

    y_test_predicted_mc = None

    j = 0

    for c1 in classes:
        for c2 in classes:
            if c1 < c2:
                j += 1
                clf = {}
                # computing svc for all unique combination of classes
                clf = sklearn.svm.SVC(**SVM_parameters, probability=True)
                clf.fit(np.concatenate((x_train_normalized[c1], x_train_normalized[c2]), axis=0), np.concatenate((y_train[c1], y_train[c2]), axis=0))

                if y_test_predicted_mc is not None:
                    y_test_predicted_mc = np.concatenate(
                        (y_test_predicted_mc, clf.predict(x_test_mc_normalized).reshape(-1, 1)), axis=1)
                else:
                    y_test_predicted_mc = clf.predict(x_test_mc_normalized).reshape(-1, 1)

    y_test_predicted_mc = np.array(y_test_predicted_mc, dtype=np.int)

    y_test_predicted_mv = np.zeros((len(y_test_predicted_mc),))

    for i, e in enumerate(y_test_predicted_mc):
        y_test_predicted_mv[i] = np.bincount(e).argmax()

    print(f_test_id)

    f = open("output/cm_multiclasses.txt", "a")
    # compute confusion matrix
    cm_multiclasses = compute_cm_multiclass(y_test_mc, y_test_predicted_mv)
    cm_multiclasses_sum += cm_multiclasses

    f.write(str(f_test_id))
    f.write('\n')
    f.write(str(cm_multiclasses))
    f.write('\n\n')
    f.close()

f = open("output/cm_multiclasses.txt", "a")
f.write("sum:")
f.write('\n')
f.write(str(cm_multiclasses_sum))
f.write('\n\n')
f.close()

for i in range(0, len(classes)):
    line_sum = (sum(cm_multiclasses_sum[i]))
    cm_multiclasses_sum[i] = np.round(cm_multiclasses_sum[i] * 100 / line_sum, 1)

total_accuracy = sum(np.diag(cm_multiclasses_sum)) / 10


f = open("output/cm_multiclasses.txt", "a")
f.write("percent:")
f.write('\n')
f.write(str(cm_multiclasses_sum))
f.write('\n\n')
f.write("total accuracy: ")
f.write(str(np.round(total_accuracy, 3)) + "%")
f.write('\n\n')
f.close()

