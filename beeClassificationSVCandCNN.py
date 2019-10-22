import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import display
%matplotlib inline

import pandas as pd
import numpy as np

from skimage import io
from skimage.feature import hog
from skimage.color import rgb2grey

#SVC and PCA imports
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

import pickle
from pathlib import Path

#Convolutional Neural Nets(Deep learning) Imports
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

labels = pd.read_csv("datasets/labels.csv", index_col=0)

def get_image(row_id, path="datasets/"):
    """
    convert the image identification number to filepath then converted
    to an array for furthre analysis
    """
    name = "{}.jpg".format(row_id)
    filepath = os.path.join(path, name)
    img = Image.open(filepath)
    return np.array(img)

#subset and display the image of an Apis
apis_row = labels[labels.genus == 0.0].index[5]
plt.imshow(get_image(apis_row))
plt.show()

#subset and display the image of a bubblebee
bombus_row = labels[labels.genus == 1.0].index[5]
plt.imshow(get_image(bombus_row))
plt.show()

#convert the image to grayscale and apply HOG
bombus = get_image(bombus_row)
grey_bombus = rgb2grey(bombus)
plt.imshow(grey_bombus, cmap=mpl.cm.gray)

#apply histogram of oriented gradients (hog) to pick out object's shape as the features were uncleared
hog_features, hog_image = hog(grey_bombus,
                              visualize=True,
                              block_norm='L2-Hys',
                              pixels_per_cell=(16, 16))
#display the hog_image
plt.imshow(hog_image, cmap=mpl.cm.gray)

# crete_features combines pixel values from image and HOG features and flatten the 3D array to 1D array

def combined_features(img):
   
    color_features = img.flatten()
    grey_scale_image = rgb2grey(img)
    hog_features = hog(grey_scale_image, block_norm='L2-Hys', pixels_per_cell=(16, 16))
    # combine these features into a single array
    combined_features = np.hstack((color_features,hog_features))
    return combined_features

bombus_features = combined_features(bombus)


#loop over all images in the dataframe
def create_feature_matrix(label_dataframe):
    features_list = []
    
    for img_id in label_dataframe.index:
        img = get_image(img_id)
        image_features = combined_features(img)
        features_list.append(image_features)
        
    # convert list of arrays into a matrix to pass into the machine learning model
    feature_matrix = np.array(features_list)
    return feature_matrix

feature_matrix = create_feature_matrix(labels)

#apply PCA to for features reduction and SVM for classification
ss = StandardScaler()
bees_stand = ss.fit_transform(feature_matrix)
pca = PCA(n_components=500)
bees_pca = pca.fit_transform(bees_stand)
#reduce from 31296 features to 500 features

#split the data and test the model on remaining 30%
#150 test photos and 350 train photos
X_train, X_test, y_train, y_test = train_test_split(bees_pca,
                                                    labels.genus.values,
                                                    test_size=.3,
                                                    random_state=1234123)
#SVC is used for classification
svm = SVC(kernel='linear',probability=True,random_state=42)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_pred,y_test)
print('Model accuracy is: ', accuracy)
#the accuracy from SVC is 68%

#normalize image for CNN
ss = StandardScaler()

image_list = []
for i in labels.index:
    img = io.imread('datasets/{}.jpg'.format(i)).astype(np.float64)
    
    # for each channel, apply standard scaler's fit_transform method
    for channel in range(img.shape[2]):
        img[:, :, channel] = ss.fit_transform(img[:, :, channel])
        
    # append to list of all images
    image_list.append(img)
    
# convert image list to single array
X = np.array(image_list)

#split the data
x_interim, x_eval, y_interim, y_eval = train_test_split(X,
                                           y,
                                           test_size=0.2,
                                           random_state=52)

#train + test = 80% and holdout set is 20%
x_train, x_test, y_train, y_test = train_test_split(x_interim,y_interim, test_size=0.4, random_state=52)

num_classes = 1

#build the deep learning model by initializing sequential and 2 convolutional layers
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(50, 50, 3)))
model.add(Conv2D(64,kernel_size = (3, 3), activation='relu'))
#reduce dimentionality by downscaling the image this case, (2,2) moving window
model.add(MaxPooling2D(pool_size=(2,2)))
#third convolutional layer to better learn features from images
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
#add dropout to prevent overfitting
model.add(Dropout(0.25))
#flatten the outputs into 1D vector
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
#use sigmoid for classifcation whether it is bumble bee or honey bee
model.add(Dense(num_classes, activation='sigmoid', name='preds'))
model.summary()

#compile the model using binary crossentropy loss function and classical stochastic gradient descent optimizer
model.compile(  
    loss=keras.losses.binary_crossentropy,
    optimizer=keras.optimizers.SGD(lr=0.001),
    metrics=['accuracy']
)
#mock-train the model with the ten observations
model.fit(
    x_train[:10, :, :, :],
    y_train[:10],
    epochs=5,
    verbose=1,
    validation_data=(x_test[:10, :, :, :], y_test[:10])
)

pretrained_cnn = keras.models.load_model('datasets/pretrained_model.h5')

#evaluate the loss and accuracy on test and holdout sets
score = pretrained_cnn.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



# evaluate model on holdout set
eval_score = pretrained_cnn.evaluate(x_eval, y_eval, verbose=0)
# print loss score
print('Eval loss:', eval_score[0])
# print accuracy score
print('Eval accuracy:', eval_score[1])
# the model received 65% accuracy which is worst than SVC in 350 trained data but 80% accuracy for 784 trained data







