from keras.utils import multi_gpu_model
from keras.applications.xception import Xception
from keras.models import Model, Sequential
from keras.layers import Input, Lambda, BatchNormalization, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

class Train:
    def __init__(self, train_segment_dir, val_segment_dir, batch_size=32, dims=(224, 224)):
        self.train_segment_dir = train_segment_dir
        self.val_segment_dir = val_segment_dir
        self.img_width = dims[1]
        self.img_height = dims[0]
        self.batch_size = batch_size
    def build_model(self):
        print("Preparing to load images and the assigned classes...")
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            rotation_range=15,
            shear_range=0.01,
            height_shift_range=0.1,
            width_shift_range=0.1,
            zoom_range=[0.9, 1.25],
            vertical_flip=False,
            horizontal_flip=True,
            brightness_range=[0.5,1.5],
            fill_mode='nearest')
        val_datagen = ImageDataGenerator(
            rescale=1.0 / 255.0)

        train_generator = train_datagen.flow_from_directory(
            self.train_segment_dir,
            target_size=(self.img_height, self.img_width),
            color_mode='rgb',
            batch_size=self.batch_size,
            class_mode='categorical')
        validation_generator = val_datagen.flow_from_directory(
            self.val_segment_dir,
            target_size=(self.img_height, self.img_width),
            color_mode='rgb',
            batch_size=self.batch_size,
            class_mode='categorical')

        print("Build the model...")

        # Use transfer learning from ResNet50 instead - discard Dense layers
        model = Xception(weights='imagenet', input_shape=(self.img_width, self.img_height, 3),
                         include_top=False)
        # Make sure to preprocess input that ResNet50 is expecting
        #inp = Input(shape=(self.img_width, self.img_height, 3))
        #x = Lambda(lambda x: preprocess_input(x))

        inp = model.input
        
        # Create new dense layers on top
        x = model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(10, activation='softmax')(x)
        final_model = Model(input=inp, outputs=predictions)

        # Only set batchnorm, dense and softmax layer to trainable
        #for layer in final_model.layers:
        #    s = layer.name.lower()
        #    layer.trainable = False
        #    if 'bn' in s or 'dense' in s or 'softmax' in s:
        #        layer.trainable = True

        final_gpu_model = multi_gpu_model(final_model, gpus=4) # New - training over multiple GPUs
        filepath = 'model_bodyfat.hdf5'
        checkpoint = ModelCheckpoint(filepath,
                                     monitor='val_acc',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='max')
        callbacks_list = [checkpoint]
        final_gpu_model.summary()
        final_gpu_model.compile(loss='categorical_crossentropy',
                                optimizer='adam',
                                metrics=['accuracy'])
        print('Training the model...')
        history = final_gpu_model.fit_generator(
            train_generator,
            steps_per_epoch=train_generator.samples // self.batch_size,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // self.batch_size,
            callbacks=callbacks_list,
            verbose=1,
            max_queue_size=10,
            workers=64,
            shuffle=False,
            epochs=250)

        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        with open('results.csv', 'w') as f:
            f.write('iter,acc,val_acc,loss,val_loss\n')
            for k, (a, va, l, vl) in enumerate(zip(acc, val_acc, loss, val_loss)):
                f.write('{},{},{},{},{}\n'.format(k + 1, a, va, l, vl))


train = Train(train_segment_dir='/home/ubuntu/travis/ImageProcessing/output_seg/train_segment',
              val_segment_dir='/home/ubuntu/travis/ImageProcessing/output_seg/val_segment',
              batch_size=16, dims=(511, 511))
train.build_model()
