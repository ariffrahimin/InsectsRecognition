# There are twoo model that is Mark1 and Mark2. 
# Mark1 is a complex layered Neural Network with over 6 Million trainable parameter
# Mark2 is a Fast-Compact Neural Network with 100 000 trainable parameter


from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras import Model

def model_mark_1(nbr_classes):
    my_input = Input(shape=(150, 150, 3))
    
    x=Conv2D(32, (3,3), activation='relu')(my_input)
                
    x=MaxPooling2D(2, 2)(x)
    x=Dropout(0.3)(x)
    x=Conv2D(64, (3,3), activation='relu')(x)
    x=MaxPooling2D(2, 2)(x)
    x=Dropout(0.3)(x)
    
    x=Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x=MaxPooling2D(2, 2)(x)
    x=Dropout(0.3)(x)
    
    x=Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x=MaxPooling2D(2, 2)(x)
    x=Dropout(0.3)(x)
    
    x=Conv2D(512, (3,3), activation='relu', padding='same')(x)
    x=MaxPooling2D(2, 2)(x)
    x=Dropout(0.3)(x)
        
    x=Conv2D(512, (3,3), activation='relu', padding='same')(x)
    x=MaxPooling2D(2, 2)(x)
    x=Dropout(0.3)(x)
    
    x=Flatten()(x)
    x=Dense(1024, activation='relu')(x)
    x=Dropout(0.5)(x)

    x=Dense(512, activation='relu')(x)
    x=Dropout(0.5)(x)
    
    x=Dense(nbr_classes, activation='softmax')(x)
    

    return Model(inputs=my_input, outputs=x)

def model_mark_2(nbr_classes):
    my_input = Input(shape=(150,150, 3))

    x=Conv2D(32, (3, 3), activation='relu')(my_input)
    x=MaxPool2D()(x)
    x=BatchNormalization()(x)
    
    x=Conv2D(64, (3, 3), activation='relu')(x)
    x=MaxPool2D()(x)
    x=BatchNormalization()(x)

    x=Conv2D(128, (3, 3), activation='relu')(x)
    x=MaxPool2D()(x)
    x=BatchNormalization()(x)

    x=GlobalAvgPool2D()(x)
    x=Dense(128, activation='relu')(x)
    x=Dense(nbr_classes, activation='softmax')(x)

    return Model(inputs=my_input, outputs=x)