from sklearn.model_selection import train_test_split
from my_utils import create_generators
from deep_learning_model import model_mark_2, model_mark_1
import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import tensorflow as tf
from ann_visualizer.visualize import ann_viz

if __name__=="__main__":


    path_to_train = "E:\\Projects\\Python\\InsectsRecognition\\data\\training_data\\train"
    path_to_val = "E:\\Projects\\Python\\InsectsRecognition\data\\training_data\\val"
    path_to_test = "E:\\Projects\\Python\\InsectsRecognition\data\\Test"
    batch_size = 64
    epochs = 100
    train_generator, val_generator, test_generator = create_generators(batch_size, path_to_train, path_to_val, path_to_test)
    nbr_classes = train_generator.num_classes

    # Change these values to choose the action
    TRAIN=True
    TEST=False
    VISUALIZE=False

    if TRAIN:
        # Choose which Deeplearning model you want to use and select the apropriate folder
        # eg: if you choose the mark1, then set the path_to_save_model to mark1 folder
        path_to_save_model = './Mark2_Model'

        ckpt_saver = ModelCheckpoint(
            path_to_save_model,
            monitor="val_accuracy",
            mode='max',
            save_best_only=True,
            save_freq='epoch',
            verbose=1
        )

        early_stop = EarlyStopping(monitor="val_accuracy", patience=10)

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tb_call = TensorBoard(log_dir=log_dir, histogram_freq=1)

        # choose either model_mark_1 or model_mark_2
        model = model_mark_2(nbr_classes)

        model.summary()

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(train_generator,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=val_generator,
                callbacks=[ckpt_saver, early_stop]
            )
    
    if TEST:
        model = tf.keras.models.load_model('./Mark2_Model')
        model.summary()
        print("Evaluating validation set : ")
        model.evaluate(val_generator)
        print("Evaluating test set : ")
        model.evaluate(test_generator)
    
    if VISUALIZE:
        model = model_mark_2(nbr_classes)
        ann_viz(model)