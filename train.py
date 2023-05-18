from keras.utils import np_utils
from utils.read_videos import Videoto3D
from utils.load_data import loaddata
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from model import TDCNNLSTM
from utils.dataset import classdata
import random
import matplotlib.pyplot as plt

#add path to folder for all video files
dir_path_all = ["drive/MyDrive/BDD/other_clipped/class0","drive/MyDrive/BDD/other_clipped/class1","drive/MyDrive/BDD/other_clipped/class2","drive/MyDrive/BDD/other_clipped/class3","drive/MyDrive/BDD/other_clipped/class4","drive/MyDrive/BDD/other_clipped/class5","drive/MyDrive/BDD/other_clipped/class6","drive/MyDrive/BDD/other_clipped/class7"]

_, class0_train, class1_train, class2_train, class3_train, class4_train, class5_train, class6_train, class7_train = classdata(dir_path_all)

random.seed(10)
class0_train = random.sample(class0_train, int(len(class0_train)/10))
class1_train = random.sample(class1_train, int(len(class1_train)/10))
class2_train = random.sample(class2_train, int(len(class2_train)/10))
class3_train = random.sample(class3_train, int(len(class3_train)/10))
class4_train = random.sample(class4_train, int(len(class4_train)/10))
class5_train = random.sample(class5_train, int(len(class5_train)/10))
class6_train = random.sample(class6_train, int(len(class6_train)/10))
class7_train = random.sample(class7_train, int(len(class7_train)/10))

training_set_10 = class0_train + class1_train + class2_train + class3_train + class4_train + class5_train + class6_train + class7_train

cmodel = TDCNNLSTM()

#parameters for dataset
depth = 16
img_rows, img_cols, frames = 112, 112, depth
channel = 3 
batch = 16
epoch = 30
# videos = '/content/drive/MyDrive/BDD/clipped_videos_4class'
color = True
skip = False
# nclass = 8

vid3d = Videoto3D(img_rows, img_cols, frames)
nb_classes = 8
#return data in x and labels in y
# for i, training_set in enumerate(training_sets):
x, y = loaddata(training_set_10, vid3d, color, skip)

print(x.shape)

#reshape x and convert labels to categorical for the model
X = x.reshape((x.shape[0], frames, img_cols, img_rows, channel))
Y = np_utils.to_categorical(y,num_classes=nb_classes)

X = X.astype('float32')
    
print('\nX_shape:{}\nY_shape:{}'.format(X.shape, Y.shape))

history = cmodel.fit(X, Y, batch_size=batch,
                epochs=30, verbose=1, shuffle=True)

#save the model 
cmodel.save('/Users/nanthininarayanan/Desktop/Courses/BDD/Code Database/models/RandomForest_8class_Model_10.h5')

plt.plot(history.history['accuracy'])
plt.title('Training Progress')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.plot(history.history['loss'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['accuracy','loss'], loc='upper right')
plt.show()