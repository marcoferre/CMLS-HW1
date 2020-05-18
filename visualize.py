import json
import matplotlib.pyplot as plt
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


f = open("output/tot_features.json", "r")
f_str = f.read()

tot_features = json.loads(f_str)

dict_train_features = tot_features[0]

for c in classes:
    arr = np.array(dict_train_features[c])
    mfcc = arr.transpose()

    # Visualization
    fig = plt.figure(figsize=(16, 6))
    plt.subplot(1,2,1)
    plt.imshow(mfcc, origin='lower', aspect='auto')
    plt.xlabel('Training samples')
    plt.ylabel('MFCC coefficients')
    plt.title('MFCC (coefficients 0 to 13) for class {}'.format(c))
    plt.colorbar()
    plt.tight_layout()

    mfcc_upper = mfcc[4:]
    plt.subplot(1,2,2)
    plt.imshow(mfcc_upper, origin='lower', aspect='auto')
    plt.title('MFCC (coefficients 4 to 13) for class {}'.format(c))
    plt.xlabel('Training samples')
    plt.ylabel('MFCC coefficients')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('plot/{}'.format(c))
    plt.show()
