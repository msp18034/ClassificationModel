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

def create_model(ing_num,classes):
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(4096, activation='relu', name="fc1")(x)
    x = Dropout(0.3)(x)
    ingredients = Dense(2048, activation='relu', name="fc2")(x)
    ingredients = Dropout(0.3)(ingredients)
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

f = open("3Classs.txt",'r')
images=[]
ingres=[]
classes=[]
for line in f:
    path,ingre=line.split(" ",1)
    classname=path.split("/")[1]
    ingre=np.fromstring(ingre, dtype=int, sep=' ')
    image=cv2.imread("image"+path)
#     image = cv2.resize(image, (224, 224))
    if classname=='3':
        classname='0'
    images.append(image)
    ingres.append(ingre)
    classes.append(classname)
images=np.array(images)
classes=np.array(classes)
y_train = keras.utils.to_categorical(classes, 3)

ingres=np.array(ingres)

model_path="model.h5"
model=create_model(353,3)
model.compile(
            loss={
                'ingredients': 'binary_crossentropy',
                'predictions': 'categorical_crossentropy'
                  },
            loss_weights={
                'ingredients': 1.,
                'predictions': 9
                },
            #optimizer='adam',
            optimizer=SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True),
            metrics=['accuracy'])
#model.fit(images,classes, batch_size=32,epochs=10)
history=model.fit(images,[ingres,y_train], batch_size=32,shuffle=True,validation_split=0.1,epochs=100)

model.save(model_path)
