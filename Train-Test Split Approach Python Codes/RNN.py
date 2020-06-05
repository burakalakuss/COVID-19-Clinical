ry addition
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

#data read
clinicalInput = pd.read_excel("..../")

#determine the number of classes(labels)
label_encoder = LabelEncoder().fit(clinicalInput.Label)
labels = label_encoder.transform(clinicalInput.Label)
classes = list(label_encoder.classes_)

#data preparation
train = train.drop(["Patient ID", "Patient age quantile","Label"],axis=1)

#determine number of features and classes
nb_features = 18
nb_classes = len(classes)

#Standardization of train data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(clinicalInput.values)
train = scaler.transform(clinicalInput.values)

#split train data into validation and train
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(clinicalInput, labels, test_size=0.2)

#determine the categories of labels
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_valid = to_categorical(y_valid)

#resize the input data
X_train = np.array(X_train).reshape(480,18,1)
X_valid = np.array(X_valid).reshape(120,18,1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv1D, Dropout, MaxPooling1D, Flatten, SimpleRNN, BatchNormalization

model = Sequential()
model.add(SimpleRNN(512,input_shape=(nb_features,1)))
model.add(Activation("relu"))
model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add((Flatten()))
model.add(Dense(2048, activation="relu"))
model.add(Dense(1024, activation="relu"))
model.add(Dense(nb_classes, activation="softmax")) 
model.summary()

from tensorflow.keras.optimizers import SGD
opt = SGD(lr=1e-3, decay=1e-5, momentum=0.3, nesterov=True)

from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

#compile the model
model.compile(loss="binary_crossentropy", optimizer = opt, metrics=["accuracy",f1_m,precision_m,recall_m])

#run the model
score = model.fit(X_train, y_train, epochs = 250, validation_data=(X_valid,y_valid))

y_score = model.predict((X_valid))

#Necessary Informations
print(("Average Training loss: ", np.mean(score.history["loss"])))
print(("Average Training Accuracy: ", np.mean(score.history["accuracy"])))
print(("Average Validation loss: ", np.mean(score.history["val_loss"])))
print(("Average Validation Accuracy: ", np.mean(score.history["val_accuracy"])))
print(("Average F1-Score: ", np.mean(score.history["val_f1_m"])))
print(("Average Precision: ", np.mean(score.history["val_precision_m"])))
print(("Average Recall: ", np.mean(score.history["val_recall_m"])))

#plot training and validation values
import matplotlib.pyplot as plt
plt.plot(model.history.history["accuracy"])
plt.plot(model.history.history["val_accuracy"])
plt.title("Model Accuracies")
plt.ylabel("Accuracy")
plt.xlabel("Number of Epochs")
plt.legend(["Train","Test"], loc="upper left")
plt.show()

from sklearn.metrics import roc_curve, auc
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(nb_classes):
    fpr[i], tpr[i], _ = roc_curve(y_valid[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_valid.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

#roc plot for specific class
plt.figure()
lw = 2
plt.plot(fpr[0], tpr[0], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('RNNROC')
plt.legend(loc="lower right")
plt.show()

from itertools import cycle
from scipy import interp

lw = 2
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nb_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(nb_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= nb_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label=''
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label=''
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(nb_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for each class')
plt.legend(loc="lower right")
plt.show()