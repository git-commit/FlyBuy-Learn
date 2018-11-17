import os
import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

import multiprocessing
import signal

# The original class can be imported like this:
# from keras.preprocessing.image import ImageDataGenerator

# We access the modified version through T.ImageDataGenerator
import tools.image as T

def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def main():
    # make sure soft-placement is off
    tf_config = tf.ConfigProto(allow_soft_placement=False)
    tf_config.gpu_options.allow_growth = True
    s = tf.Session(config=tf_config)
    K.set_session(s)

    # parameters
    model_name = "models/inception_kaufland.h5"
    num_classes = 12
    size = (299, 299)
    in_size = 3600
    bs = 12
    steps_per_epoch = in_size // bs

    val_size = 1200
    steps_val = val_size // bs

    cores = 8
    pool = multiprocessing.Pool(processes=cores, initializer=init_worker)

    train_datagen = T.ImageDataGenerator(
     rescale=1./255,
     rotation_range=180,
     width_shift_range=.2,
     height_shift_range=.2,
     #brightness_range=(0.5, 1),
     shear_range=0.2,
     zoom_range=0.2,
     fill_mode="nearest",
     pool=pool)

    train_generator = train_datagen.flow_from_directory(
            'data/train_set_xs',
            target_size=size,
            batch_size=bs,
            class_mode='categorical',
            #interpolation="bicubic"
            )

    test_datagen = T.ImageDataGenerator(
         rescale=1./255,
         pool=pool
         )

    test_generator = test_datagen.flow_from_directory(
            'data/test_set',
            target_size=size,
            batch_size=bs,
            class_mode='categorical')


    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have `num_classes` classes
    predictions = Dense(num_classes, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop',
     loss='categorical_crossentropy',
     metrics=['acc'])

    # train the model on the new data for a few epochs
    model.fit_generator(train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=10,
            validation_data=test_generator,
            validation_steps=steps_val)

    # at this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from inception V3. We will freeze the bottom N layers
    # and train the remaining top layers.

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    #for i, layer in enumerate(base_model.layers):
    #   print(i, layer.name)

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in model.layers[:249]:
       layer.trainable = False
    for layer in model.layers[249:]:
       layer.trainable = True

    print("Fine tuning...")
    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    from keras.optimizers import SGD
    model.compile(
        optimizer=SGD(lr=0.0001, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['acc']
        )

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    model.fit_generator(train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=10,
            validation_data=test_generator,
            validation_steps=steps_val)

    model.save(os.path.join(os.getcwd(), model_name))

if __name__ == '__main__':
    main()
