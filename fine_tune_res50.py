from keras.applications.resnet50 import ResNet50
from keras.layers import Input
from keras.optimizers import SGD
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model,load_model
from keras.layers import Dense, GlobalAveragePooling2D,Dropout,Flatten,BatchNormalization
from keras import backend as K
import keras
import numpy as np
import argparse
import math
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import pickle

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--epoch', type=int,
                    help='set input ecpoch')
parser.add_argument('--baseModel', type=int,
                    help='mobile,inceptionV3 or Resnet50')

parser.add_argument('--reload', type=int,
                    help='reload or train from 0')
parser.add_argument('--train', type=int,
                    help='train 1,evaluate 0 or whatever')

parser.add_argument('--model', type=str,
                    help='model path to save or reload')


args = parser.parse_args()

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

def read_val_img(path, target_size):
    try:
        img = Image.open(path).convert("RGB")
        img_rs = img.resize(target_size)
#       img_rs = img_rs*1.0/255
    except Exception as e:
        print(e)
    else:
        x =np.array(img_rs)
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
    #base_model = InceptionV3(weights='imagenet', include_top=False)
    Inp = Input((256, 256, 3))
    if args.baseModel=="inceptionV3":
        base_model = InceptionV3(weights='imagenet', include_top=False)
    elif args.baseModel=="resnet50":
        base_model = ResNet50(weights='imagenet', include_top=False,
                              input_shape=(256, 256, 3), )
    elif args.baseModel=="mobilenet":
        base_model = keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet', include_top=False)

    K.set_learning_phase(1)
    x = base_model(Inp)
    x = BatchNormalization(axis=-1, name='banNew')(x)
    x = GlobalAveragePooling2D(name='average_pool')(x)

    ingredients = Dense(1024, activation='relu', name="fc2")(x)
    ingredients = Dropout(0.5)(ingredients)
    ingredients = Dense(ing_num, activation='sigmoid', name="ingredients")(ingredients)

    predictions = Dense(4096, activation='relu', name="fc3")(x)
    predictions = Dropout(0.5)(predictions)
    predictions = Dense(classes, activation='softmax', name="predictions")(predictions)

    model = Model(input=Inp, output=[ingredients, predictions])
    for layer in base_model.layers:
        layer.trainable = False
    return model


def my_gen(path,nbclass, batch_size, target_size,mode):
    img_list =getList(path)
    #steps=500
    steps = math.ceil(len(img_list) / batch_size)
    print("Found %s images."%len(img_list))
    while True:
        for i in range(steps):
            batch_list = img_list[i * batch_size : i * batch_size + batch_size]
    
            x=[parse_path(line) for line in batch_list]
            x = [read_img(file, target_size) for file in x]
            batch_x = np.concatenate([array for array in x])
            y = [ parse_label(line) for line in batch_list]
            batch_y= keras.utils.to_categorical(y,nbclass)
            
            ingres = [ parse_ingres(line) for line in batch_list]
            batch_ingres= np.concatenate([array for array in ingres])  
            if(mode=="train"):
                a = image_gen.flow(batch_x,batch_y,batch_size = batch_x.shape[0],shuffle=False)
                x,y=next(a)
                yield x,[batch_ingres,y]
            else:
              #  a = valid_gen.flow(batch_x,batch_y,batch_size = batch_x.shape[0],shuffle=False)
               # x,y=next(a)
                #print("image after ---------------",x[0])
                batch_x=batch_x*1.0/255
                yield batch_x,[batch_ingres,batch_y]

train_path = "train.txt"
val_path = "val.txt"
test_path = 'test.txt'
batch_size = 64
nbclass = 173
steps = math.ceil(len(getList(train_path)) / batch_size)
val_steps = math.ceil(len(getList(val_path)) / batch_size)
test_steps = math.ceil(len(getList(test_path)) / batch_size)

target_size = (256,256)

image_gen=ImageDataGenerator(rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

train_gen =my_gen(train_path,173, batch_size, target_size,"train")
val_gen=my_gen(val_path,173, batch_size, target_size,"val")
test_gen = my_gen(test_path, 173, batch_size, target_size, "val")

model_path=args.model

if args.reload==0:
    #odel_path="model.h5"
    model=create_model(353,nbclass)
    model.compile(
            loss={
                'ingredients': 'binary_crossentropy',
                'predictions': 'categorical_crossentropy'
                  },
            loss_weights={
                'ingredients': 0.1,
                'predictions': 0.9
                },
            #optimizer='adam',
            optimizer=SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True),

            metrics=['accuracy','top_k_categorical_accuracy'])
else:
    model=keras.models.load_model(model_path)

if args.train==1: 
    history=model.fit_generator(generator=train_gen, steps_per_epoch=steps, epochs=args.epoch,validation_data=val_gen,validation_steps=val_steps, verbose=1,use_multiprocessing=True, workers=1)
    model.save(model_path)

    with open('lr001'+str(args.epoch), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


else: # Evaluate
    history = model.evaluate_generator(test_gen, steps=test_steps, verbose=1)
    print(history[3])
    print(history[4])
    print(history[5])
    print(history[6])




