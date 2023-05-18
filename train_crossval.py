from keras.utils import np_utils
from utils.read_videos import Videoto3D
from utils.load_data import loaddata
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from model import TDCNNLSTM
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report,confusion_matrix


#parameters for dataset
depth = 16
img_rows, img_cols, frames = 112, 112, depth
channel = 3 
batch = 16
epoch = 30
videos = '/content/drive/MyDrive/BDD/clipped_videos'
color = True
skip = False
nclass = 8

vid3d = Videoto3D(img_rows, img_cols, frames)
nb_classes = 8
#return data in x and labels in y
x, y = loaddata(videos, vid3d, nclass, color, skip)

print(x.shape)

#reshape x and convert labels to categorical for the model
X = x.reshape((x.shape[0], frames, img_cols, img_rows, channel))
Y = np_utils.to_categorical(y,num_classes=nb_classes)

X = X.astype('float32')
    
print('\nX_shape:{}\nY_shape:{}'.format(X.shape, Y.shape))

#perform k-fold cross validation
scores, histories = list(), list()
Acc = []
preds = []
labs = []
predScore = []
kfold = StratifiedKFold(5, shuffle=True, random_state=1)
# enumerate splits
k = 0

for train_ix, test_ix in tqdm(kfold.split(X,np.argmax(Y, axis=1))):
    print('kfold {}'.format(k+1)) 
    # select rows for train and test
    trainX, trainY, testX, testY = X[train_ix], Y[train_ix], X[test_ix], Y[test_ix]
    cmodel = TDCNNLSTM()
    # fit model
    history = cmodel.fit(trainX, trainY, batch_size=batch,
                    epochs=30, verbose=1, shuffle=True)
    
    histories.append(history)
    plt.plot(history.history['accuracy'])
    plt.title('Training Progress')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.plot(history.history['loss'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['accuracy','loss'], loc='upper right')
    plt.show()
    # evaluate model
    _, acc = cmodel.evaluate(testX, testY, verbose=0)
    # append scores
    scores.append(acc)
    #print('Accuracy: %.3f%% (+/- %.3f)' % (np.mean(scores)*100, np.std(scores)))
    rounded_predictions = cmodel.predict(testX)
    score = rounded_predictions
    rounded_predictions = np.argmax(np.round(rounded_predictions),axis=1)
    preds.extend(rounded_predictions)
    predScore.extend(score)
    labs.extend(testY)
    rounded_labels=np.argmax(testY, axis=1)
    Acc.append(np.count_nonzero(rounded_labels==rounded_predictions)/len(rounded_predictions)*100)
    k = k + 1
    
#Evaluate performance 
rounded_labels=np.argmax(labs, axis=1)
print(classification_report(rounded_labels, preds))
Acc= (np.count_nonzero(rounded_labels==preds))/len(preds)*100
print('Acc:',Acc)
#Display confusion matrix
cm = confusion_matrix(rounded_labels, preds)
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

print(scores)
#accuracy
print(Acc)
print('Cross Validation Accuracy:' , np.array(scores).mean()*100)