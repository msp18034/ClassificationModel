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
import tensorflow as tf
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
def read():
    f = open("/home/hduser/shuffled.txt",'r')
    images=[]
    ingres=[]
    classes=[]
    count=0
    for line in f:
        path,ingre=line.split(" ",1)
        classname=path.split("/")[1]
        ingre=np.fromstring(ingre, dtype=int, sep=' ')
        image=cv2.imread("/home/hduser/Vireo"+path)
        image = cv2.resize(image, (256, 256))
        count+=1
        if classname=='172':
            classname=='0'
        images.append(image)
        ingres.append(ingre)
        classes.append(classname)
        if count%1200==0:
            break
    images=np.array(images)
    classes=np.array(classes)
    y_train = keras.utils.to_categorical(classes,172)
    ingres=np.array(ingres)
    print(ingres.shape)
    return images,y_train,ingres

def _parse_record(bytestr):
    example = tf.train.Example()
    example.ParseFromString(bytestr)
    features = example.features.feature
    label = np.array(features['label'].int64_list.value)
    shape = np.array(features['shape'].int64_list.value)[0]
    image = np.array(features['image'].int64_list.value).reshape(shape)
    ingres = np.array(features['ingredients'].int64_list.value)
    return image, label,ingres

def parse_data(record):
    keys_to_features = {
        "image": tf.FixedLenFeature((), tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
        "label": tf.FixedLenFeature((), tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
        "ingres": tf.FixedLenFeature((), tf.int64,
                                    default_value=tf.zeros([], dtype=tf.int64)),
    }
    parsed = tf.parse_single_example(record, keys_to_features)

    # Parse the string into an array of pixels corresponding to the image
    images = tf.cast(parsed["image"],tf.uint8)
    print(images)
    images = tf.reshape(images,[256,256,3])
    labels = tf.cast(parsed['label'], tf.int32)
    labels = tf.one_hot(labels,172)
    ingres = tf.cast(parsed['ingres'], tf.int32)
    return images,labels,ingres
dataset=tf.data.TFRecordDataset(filenames="shuffled.tfrecord")
ds_train=dataset.map(parse_data)


model_path="model.h5"
model=create_model(353,172)
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
history=model.fit(ds_train, batch_size=32,validation_split=0.1,epochs=100)

model.save(model_path)
