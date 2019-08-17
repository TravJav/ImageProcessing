from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

class Train:
    def __init__(self, train_segment_dir, val_segment_dir, dims=(512, 512)):
        self.train_segment_dir = train_segment_dir
        self.val_segment_dir = val_segment_dir
        self.img_width = dims[1]
        self.img_height = dims[0]
    def build_model(self):
        print("Preparing to load images and the assigned classes...")
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            shear_range=0.2,
            zoom_range=0.2,
            height_shift_range=0.1,
            width_shift_range=0.1,
            vertical_flip=False,
            horizontal_flip=True)
        val_datagen = ImageDataGenerator(
            rescale=1.0 / 255.0)

        train_generator = train_datagen.flow_from_directory(
            self.train_segment_dir,
            target_size=(self.img_height, self.img_width),
            color_mode='rgb',
            batch_size=16,
            class_mode='categorical')
        validation_generator = val_datagen.flow_from_directory(
            self.val_segment_dir,
            target_size=(self.img_height, self.img_width),
            color_mode='rgb',
            batch_size=16,
            class_mode='categorical')

        print("Build the model...")
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding="same",
            input_shape=(self.img_height, self.img_width, 3), activation="relu"))
        model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
        model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
        model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
        model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))        
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))

        filepath = 'model_bodyfat.hdf5'
        checkpoint = ModelCheckpoint(filepath,
                                     monitor='val_acc',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='max')
        callbacks_list = [checkpoint]
        model.summary()
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        print('Training the model...')
        history = model.fit_generator(
            train_generator,
            steps_per_epoch=train_generator.samples // 16,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // 16,
            callbacks=callbacks_list,
            verbose=1,
            max_queue_size=10,
            workers=32,
            shuffle=False,
            epochs=100)

        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        with open('results.csv', 'w') as f:
            f.write('iter,acc,val_acc,loss,val_loss\n')
            for k, (a, va, l, vl) in enumerate(zip(acc, val_acc, loss, val_loss)):
                f.write('{},{},{},{},{}\n'.format(k + 1, a, va, l, vl))


train = Train(train_segment_dir='/home/ubuntu/travis/ImageProcessing/output_seg/train_segment',
              val_segment_dir='/home/ubuntu/travis/ImageProcessing/output_seg/val_segment')
train.build_model()
