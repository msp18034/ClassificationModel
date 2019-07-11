import cv2
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input
from keras.optimizers import SGD
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model,load_model
from keras.layers import Dense, GlobalAveragePooling2D,Dropout
from keras import backend as K
import keras
import numpy as np
import argparse
import math
from PIL import Image
#parser = argparse.ArgumentParser(description='Process some integers.')
#parser.add_argument('count', type=int,
                    #help='an integer for the accumulator')

#args = parser.parse_args()
def read_img(path, target_size):
    try:
        img = Image.open(path).convert("RGB")
        img_rs = img.resize(target_size)
#        img_rs = img_rs*1.0/255
    except Exception as e:
        print(e)
    else:
        x = np.expand_dims(np.array(img_rs), axis=0)
        return x

def getList(path):
    lines=[]
    with open(path, 'r') as infile:
        for line in infile:
            if line is not None:
                lines.append(line)
    return lines

def parse_path(line):
    '''
    Given a line from the training/test txt file, return parsed info.'''
    path,ingre=line.split(" ",1)
    return "VireoFood172"+path

def parse_label(line):
    '''
    Given a line from the training/test txt file, return parsed info.'''
    path,ingre=line.split(" ",1)
    classname=path.split("/")[1]
    
    return np.array(classname)

def parse_ingres(line):
    '''
    Given a line from the training/test txt file, return parsed info.'''
    path,ingre=line.split(" ",1)
    ingre=np.fromstring(ingre, dtype=int, sep=' ')
    return np.expand_dims(ingre, axis=0)

def create_model(ing_num,classes):
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(4096, activation='relu', name="fc1")(x)
    x = Dropout(0.5)(x)
    ingredients = Dense(2048, activation='relu', name="fc2")(x)
    ingredients = Dropout(0.5)(ingredients)
    ingredients = Dense(ing_num, activation='sigmoid', name="ingredients")(ingredients)

    #merged_vector = keras.layers.concatenate([x, ingredients], axis=-1)
    predictions = Dense(2048, activation='relu', name="fc3")(x)
    predictions = Dropout(0.5)(predictions)
    predictions = Dense(classes, activation='softmax', name="predictions")(predictions)

    input_tensor = Input(shape=(400, 400, 3))  # this assumes K.image_data_format() == 'channels_last'
    model = Model(input=base_model.input, output=[ingredients, predictions])       
 #  model = Model(input=base_model.input, output=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    return model


def my_gen(path,nbclass, batch_size, target_size):
    img_list =getList(path)
    steps = math.ceil(len(img_list) / batch_size)
    print("Found %s images."%len(img_list))
    while True:
        for i in range(steps):
            batch_list = img_list[i * batch_size : i * batch_size + batch_size]
    
            x=[parse_path(line) for line in batch_list]
            x = [read_img(file, target_size) for file in x]
            batch_x = np.concatenate([array for array in x])
            batch_x=batch_x*1.0/255 
            y = [ parse_label(line) for line in batch_list]
            batch_y= keras.utils.to_categorical(y,nbclass)
            
            ingres = [ parse_ingres(line) for line in batch_list]
            batch_ingres= np.concatenate([array for array in ingres])            
            yield batch_x,[batch_ingres,batch_y]

#images,y_train,ingres=read()
train_path="train.txt"
val_path="val.txt"
batch_size=64
nbclass=173
steps=math.ceil(len(getList(train_path)) / batch_size)
val_steps=math.ceil(len(getList(val_path)) / batch_size)
target_size = (224, 224)
train_gen = my_gen(train_path,173, batch_size, target_size)
val_gen=my_gen(val_path,173,batch_size,target_size)
model_path="model.h5"
model=keras.models.load_model("model.h5")


image=read_img("VireoFood172/1/10_0.jpg", target_size)
image=image*1./255

ingres,labels=model.predict(image,verbose=1)
print(np.argmax(labels))
#odeZZl.save(model_path)
