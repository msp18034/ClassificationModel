from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import pickle
import keras
# 构建不带分类器的预训练模型
base_model = InceptionV3(weights='imagenet', include_top=False)

# 添加全局平均池化层
x = base_model.output
x = GlobalAveragePooling2D()(x)

# 添加一个全连接层
x = Dense(4096, activation='relu')(x)
x=
# 添加一个分类器，假设我们有200个类
predictions = Dense(172, activation='softmax')(x)

# 构建我们需要训练的完整模型
#model = Model(inputs=base_model.input, outputs=predictions)

# 首先，我们只训练顶部的几层（随机初始化的层）
# 锁住所有 InceptionV3 的卷积层
for layer in base_model.layers:
    layer.trainable = False

model=keras.models.load_model("singleInception.h5")

# 编译模型（一定要在锁层以后操作）
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['acc'])
train_dir="Vireo/train"
validation_dir="Vireo/val"
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(256, 256),
        batch_size=128,
        class_mode='categorical') #we only have two classes

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(256,256),
        batch_size=64,
        class_mode='categorical')
history = model.fit_generator(
      train_generator,
      steps_per_epoch=2,
      epochs=2,
      validation_data=validation_generator,
      verbose=1,
      validation_steps=1)
model.save("singleInception.h5")
with open('trainsingle', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

