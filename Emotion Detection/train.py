# import all library required
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl

# style your matplotlib
mpl.style.use("seaborn-darkgrid")
# run this block

from tqdm import tqdm  # show progress bar of for loop

# list of files in train folder
files = os.listdir("./mma/MMAFEDB/train/")
files


# now create image and label array
image_array = []  # it's a list later i will convert it to array
label_array = []
path = "./mma/MMAFEDB/train/"
# loop through each sub-folder in train
for i in range(len(files)):
    # files in sub-folder
    file_sub = os.listdir(path + files[i])

    # print(len(file_sub))
    # loop through each files

    # for neutral and happy dataset we will use only 18000 image
    if files[i] == "neutral" or files[i] == "happy":
        for k in tqdm(range(20000)):
            # read image
            img = cv2.imread(path + files[i] + "/" + file_sub[k])
            # convert image from BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # append image_array with img
            image_array.append(img)
            label_array.append(i)
            # i is interger from 0-6
            # run this block
    else:
        # for other all
        for k in tqdm(range(len(file_sub))):
            # read image
            img = cv2.imread(path + files[i] + "/" + file_sub[k])
            # convert image from BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # append image_array with img
            image_array.append(img)
            label_array.append(i)
            # i is interger from 0-6
            # run this block

a, b = np.unique(label_array, return_counts="True")
a
import gc

gc.collect()


# now divide image_array by 255.0
# this wil scale image pixel from 0-255 to 0-1
image_array = np.array(image_array) / 255.0
# convert label list to array
label_array = np.array(label_array)
# run this block
# now define label_to_text
# ['surprise', 'fear', 'angry', 'neutral', 'sad', 'disgust', 'happy']
label_to_text = {
    0: "surprise",
    1: "fear",
    2: "angry",
    3: "neutral",
    4: "sad",
    5: "disgust",
    6: "happy",
}


from sklearn.model_selection import train_test_split

image_array, X_test, Y_train, Y_test = train_test_split(
    image_array, label_array, test_size=0.1
)
# you can change test size
# we are using 10% for validation

# now before running this block change X_train to image_array to save ram memory
gc.collect()


# if you want to see image and label
# define dic for converting label to test_label
# ['surprise', 'fear', 'angry', 'neutral', 'sad', 'disgust', 'happy']
label_to_text = {
    0: "surprise",
    1: "fear",
    2: "angry",
    3: "neutral",
    4: "sad",
    5: "disgust",
    6: "happy",
}


# now we will start with our model
# import all library required for model
from keras import layers, callbacks, utils, applications, optimizers
from keras.models import Sequential, Model, load_model

# run this block


model = Sequential()
# I will use MobileNetV2 as an pretrained model
pretrained_model = applications.mobilenet_v2.MobileNetV2(
    input_shape=(48, 48, 3), include_top=False, weights="imagenet"
)
# you can use other pretrained model to increase accuracy or increase frame rate
# change all non-trainable layer to trainable
pretrained_model.trainable = True
# add pretrained_model to model
model.add(pretrained_model)
model.add(layers.GlobalAveragePooling2D())
# add dropout to increase accuracy by not overfitting
model.add(layers.Dropout(0.3))
# add dense layer as final output
model.add(layers.Dense(1))
model.summary()


from tensorflow.keras.optimizers import Adam

# compile model

model.compile(optimizer=Adam(0.0001), loss="mean_squared_error", metrics=["mae"])
# run
# starting learning rate is 1e-3
# you can change optimizer, loss function, metrics for better result
ckp_path = "trained_model/model"
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=ckp_path,
    monitor="val_mae",
    save_best_only=True,
    save_weights_only=True,
    mode="auto",
)
# this checkpoint save model when val_mae is lower then best val_mae
# run
# now we will define learning rate reducer
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    factor=0.9,
    monitor="val_mae",
    mode="auto",
    cooldown=0,
    patience=5,
    verbose=1,
    min_lr=1e-6,
)
# this will decrease learning rate when val_mae does't decrease durning last 5 epoch
# verbose is use to show val_mae every epoch
EPOCHS = 300
BATCH_SIZE = 64

# start training
history = model.fit(
    image_array,
    Y_train,
    validation_data=(X_test, Y_test),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[model_checkpoint, reduce_lr],
)
# run

# after training is finished
# load best model
model.load_weights(ckp_path)

# if you want to see result
prediction_val=model.predict(X_test,batch_size=BATCH_SIZE)

# prediction value 
prediction_val[:10]

# original value
Y_test[:10]

# now convert model to tensorflow lite model 
converter=tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model=converter.convert()

#save model 
with open("model.tflite","wb") as f:
    f.write(tflite_model)