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



def parse_label(line):
    '''
    Given a line from the training/test txt file, return parsed info.'''
    path,ingre=line.split(" ",1)
    classname=path.split("/")[1]
    return "VireoFood172"+path,classname

def parse_ingres(line):
    '''
    Given a line from the training/test txt file, return parsed info.'''
    path,ingre=line.split(" ",1)
    ingre=np.fromstring(ingre, dtype=int, sep=' ')
    return np.expand_dims(ingre, axis=0)

def createDataframe(filePath):
    paths=[]
    labels=[]
    with open("full.txt", 'r') as infile:
        for line in infile:
            path,label=parse_label(line)
            paths.append(path)
            labels.append(label)
    diction={'paths':paths,'labels':labels}
            
    return pd.DataFrame(diction)

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
    #model = Model(input=base_model.input, output=[ingredients, predictions])       
    model = Model(input=base_model.input, output=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    return model


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
model=create_model(353,nbclass)
model.compile(
            loss={
                'ingredients': 'binary_crossentropy',
                'predictions': 'categorical_crossentropy'
                  },
            loss_weights={
                'ingredients': 2.,
                'predictions': 4
                },
            #optimizer='adam',
            optimizer=SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True),
            metrics=['accuracy'])
#model.fit(images,classes, batch_size=32,epochs=10)
#history=model.fit(images,[ingres,y_train], batch_size=32,validation_split=0.1,epochs=100)
model.fit_generator(generator=train_gen, steps_per_epoch=steps, epochs=75, verbose=1,validation_data=val_gen,validation_steps=200,
                    use_multiprocessing=True, workers=1)

model.save(model_path)
