# CMLS-HW1
Assignment 4: Audio Event Classification - Samuele Bosi, Marco Ferrè, Philip Michael Grasselli, Simone Mariani 

### 1. Introduction
In this homework our task was to implement a classiﬁer able to predict the audio event recorded in an audio excerpt. Inordertodothis,wewereprovidedwithacollectionofsampledsoundsofacrowdedurbanarea,each one corresponding to a given class among a group of 10. On the top of that, those audio ﬁles were also split into 10 different folders, and heterogeneously chosen with respect to the belonging classes. Our main task was to analyze the different folders following precise criteria in order to create a model which – for any chosen testing folder – recognizes for each ﬁle its belonging class, by means of performing tests on the signal characteristics along to some feature selection methods appropriately chosen by us.

### 2. Analyzing the Data
For this process, as suggested by the author of the sampled audio, we proceeded to apply a 10-foldcross validation method: in a nutshell, this consists in selecting one fold for testing and using the remaining nine for training; after 10 iterations, corresponding to all possible combinations of testing folds, we get a pretty homogeneous analysis of the given data. We chose to use the **MFCC (Mel-Frequency Cepstral Coefﬁcient)** as our main feature selection, where: 
- all audio ﬁles are windowed by applying a Hamming window made of 1,024 samples, and a hopsize of 512 samples; 
- the STFT (Short-Time Fourier Transform) was computed for each audio ﬁle, and we ended up taking into consideration only the absolute values of the result;
- we eventually created the **Mel ﬁlter**, with n_mels=40, f_min=133.33 Hz, and f_max=6853.8 Hz, and we applied it to the modulus of the previously calculated STFT: all the computation is thus split into n_mfcc=13 blocks.

We appended all the obtained results for each audio ﬁle, belonging to each class, and enclosed in each fold, into a dictionary: for the class air_conditioner, as an example, a representation is given in Figure 1.

### 3. Classiﬁcation by Means of Support Vector Machine (SVM)
To perform the classiﬁcation, we relied on the Support Vector Machine approach. Our ﬁnal goal is to produce a **confusion matrix** for every iteration, where a different fold was taken as the test one.

#### 3.1 Motivation of the Features Choice
In the literature we can appreciate many different audio spectral descriptors, such as the Audio Spectrum Centroid, the Harmonic Ratio, etc. In our case, the Mel-Frequency Cepstrum Coefﬁcients algorithm is the most appropriate for its accuracy in sampling the speech frequency band. This method is based on triangular ﬁlters whose center is placed accordingly to the mel scale.

Moreover, if we analyzed all the features, we would have a dramatic computational cost: we opted for a compromise in order to maintain the number of the considered test folds almost untouched. This is why we just considered only the MFCC as main parameter to pursue our task.

#### 3.2 The Dictionary 

In order to accomplish the ﬁrst task, we initially created, for each fold, a `tot_train_features` dictionary, starting from the `tot_features` calculated previously by removing from the latter one the features referring to the test fold of the corresponding iteration – such that we have only the features linked to 9 folders of each of the 10 iterations. Then, for each folder, we loaded into two vectors the feature values of each audio ﬁle, and we created four vectors divided by class: 
- two of them would contain all the feature values for the training set, and the testing set, respectively, divided by class;
-  the other two would include the correct values to be associated to each element belonging to the relative class (e.g.: all the elements associated to the class air_conditioner would have the 0 value, car_horn elements would all have value 1, etc.). As for the two vectors previously itemized, we had also here one vector for the training set, and the other for the testing set.

#### 3.3 Normalization
We could calculate, in this phase, the maximum and the minimum values from all the training sets to perform the normalization process on the training and testing sets, which is needed to perform the SVM process without unbalances, as following: 

```
X_train_normalized = (X_train−feat_min)/(feat_max−feat_min) 
X_test_normalized = (X_test−feat_min)/(feat_max−feat_min)
```

It is very important to underline the fact we used the same maximum and minimum values obtained from the training set also to normalize the testing set: this is done to ensure that, in a standard situation in which we didn’t know which audio ﬁle would be given as input to the system, the model would be already created and it should have provided any way appreciable results even for new and never-before processed inputs.

#### 3.4 Exploiting the Support Vector Machine Classiﬁer
We, then, proceeded towards the actual classiﬁcation by exploiting the **Support Vector Machine Classiﬁer** taken from the `sklearn` library. Starting off, we created a _N×N_ size dataframe using the pandas library (with _N_ being the number of the classes), that would ﬁt the entire classiﬁcation data of each one of the binary confrontation between the feature values of each class. We ﬁlled this model by putting in the corresponding cell to the concatenation of the classes, taken by couples, the SVC ﬁtting result of the concatenation of the normalized train elements of the two classes which are in analysis with the concatenation of the corresponding ideal output vectors of the same two ones. Then we append the result freshly obtained to the vector `y_test_predicted_mc`, and normalizing the values between−1 and 1. Finally, we performed a majority voting in order to obtain the `y_test_predicted_mv` vector, which we would use to calculate the multi-class confusion matrix for each one of the ten iterations. The call to this last method would provide also the `y_test_mc vector`, also known as the vector containing all the correct values that the process should have calculated during the classiﬁcation process.

#### 3.5 Final Results and Accuracy
We eventually made a **confusion matrix** related to the accuracy related to all folds, with the percentage of samples recognized by our algorithm, rounded off to two decimals. We gain the ﬁnal accuracy by calculating the mean value of the main diagonal values: Accuracy 44.0%.

In `Solution.pdf` the detailed paper.
