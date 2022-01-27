# There are twoo model that is Mark1 and Mark2. 
# Mark1 is a complex layered Neural Network with over 6 Million trainable parameter
# Mark2 is a Fast-Compact Neural Network with 100 000 trainable parameter


from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D, Flatten, Dropout, MaxPooling2D
from keras.models import Sequential


def model_mark_1(nbr_classes):
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu',
                input_shape=(150, 150, 3), padding='same'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.3))
    
    
    model.add(Conv2D(64, (3,3), activation='relu',
                input_shape=(150, 150, 3), padding='same'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.3))
        
    model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.3))
    
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(nbr_classes, activation='softmax'))

    return model  


def model_mark_2(nbr_classes):

    model = Sequential()

    model.add(Conv2D(
            32,(3, 3),
            input_shape=(150,150,3),
            activation='relu',
        ))
    model.add(MaxPool2D(2, 2))
    model.add(BatchNormalization())

    model.add(Conv2D(
            64,(3, 3),
            input_shape=(150,150,3),
            activation='relu',
        ))
    model.add(MaxPool2D(2, 2))
    model.add(BatchNormalization())
    
    model.add(Conv2D(
        128,(3, 3),
        activation='relu',
        input_shape=(150,150,3)
    ))
    model.add(MaxPool2D(2, 2))
    model.add(BatchNormalization())
    model.add(GlobalAvgPool2D())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(nbr_classes, activation='softmax'))

    return model

