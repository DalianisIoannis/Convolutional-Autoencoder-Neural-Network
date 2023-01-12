import  argparse
import  sys, os
import  numpy      as np
import  tensorflow as tf
import  keras
import  matplotlib.pyplot as plt
from    keras import layers, optimizers, losses, metrics , callbacks , models , Sequential
from    sklearn.utils import shuffle
from    sklearn.model_selection import train_test_split
from    PIL import Image
from    sklearn.metrics import precision_score,recall_score,f1_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def parse_file(FileSetName):
    images = []
    with open(FileSetName, "rb") as f:
        magic_number        = int.from_bytes(f.read(4),byteorder='big')
        number_images       = int.from_bytes(f.read(4),byteorder='big')
        number_of_rows      = int.from_bytes(f.read(4),byteorder='big')
        number_of_columns   = int.from_bytes(f.read(4),byteorder='big')
        # showIMS = 0
        for _ in range(0, number_images):
            npBytes = np.fromfile(f, dtype='B', count=number_of_columns * number_of_rows, sep='', offset=0)
            newarr = npBytes.reshape(28, 28)
            # if showIMS<2:
            #   img = Image.fromarray(newarr, 'L')
            #   plt.figure()
            #   plt.imshow(img) 
            #   plt.show()
            #   showIMS += 1
            # images.append(npBytes)
            images.append(newarr)
        npimages = np.asarray(images)
        print("magic_number", magic_number)
        print("number_images", number_images)
        print("number_of_rows", number_of_rows)
        print("number_of_columns", number_of_columns)
        f.close()
        return (npimages, magic_number, number_images, number_of_rows, number_of_columns)

def read_labels_file(filename):
  with open(filename, "rb") as f:
    _               = int.from_bytes(f.read(4),byteorder='big') # magic_number
    number_items    = int.from_bytes(f.read(4),byteorder='big')
    npBytes         = np.fromfile(f, dtype='B',count=number_items, sep='', offset=0)
    return (npBytes)

if __name__ == "__main__":
    # python classification.py -d ./train-images.idx3-ubyte -d1 ./train-labels.idx1-ubyte -t ./t10k-images.idx3-ubyte -t1 ./t10k-labels.idx1-ubyte -model ./model.h5
    with tf.device("gpu:0"):

        parser = argparse.ArgumentParser(description='Insert files')
        parser.add_argument("-d",       "--TrainingSet",    help="a training set",      required="yes")
        parser.add_argument("-d1",      "--TrainingLabels", help="a training label",    required="yes")
        parser.add_argument("-t",       "--TestSet",        help="a test set",          required="yes")
        parser.add_argument("-t1",      "--TesLabels",      help="a test label",        required="yes")
        parser.add_argument("-model",   "--EncoderModel",   help="autoencoder model",   required="yes")
        
        args            = parser.parse_args()
        TrainingSet     = args.TrainingSet
        TrainingLabels  = args.TrainingLabels
        TestSet         = args.TestSet
        TesLabels       = args.TesLabels
        EncoderModel    = args.EncoderModel

        print( "\nTrainingSet {}".format(TrainingSet))
        print( "\nTrainingLabels {}".format(TrainingLabels))
        print( "\nTestSet {}".format(TestSet))
        print( "\nTesLabels {}".format(TesLabels))
        print( "\nEncoderModel {}".format(EncoderModel))

        train_data, mnum,       nimg,       nrow,       ncol        = parse_file( TrainingSet )
        test_data,  magicNum,   numbImages, numbRows,   numbCols    = parse_file( TestSet )
        train_labels    = read_labels_file( TrainingLabels )
        test_labels     = read_labels_file( TesLabels )

        saved_model     = models.load_model( EncoderModel )
        dense_num       = 128
        lrate           = 0.001
        lrate_2         = 0.0001
        batch_sz        = 128
        Epochs_Num      = 100

        test_data   = test_data / 255
        train_data  = train_data / 255

        unknown_X_TEST  = test_data[-10:]
        unknown_y_TEST  = test_labels[-10:]
        test_data       = test_data[:-10]
        test_labels     = test_labels[:-10]

        onehot_train_labels     = np.zeros((train_labels.size,train_labels.max()+1))
        onehot_test_labels      = np.zeros((test_labels.size, test_labels.max()+1))
        onehot_train_labels[ np.arange(train_labels.size),train_labels ]  = 1
        onehot_test_labels[ np.arange(test_labels.size),test_labels ]     = 1
        
        print("\ntest_labels[0]", test_labels[0])
        print("len(test_labels)", len(test_labels))
        print("len(test_data)", len(test_data))

        classification_model    = Sequential()
        length                  = len(saved_model.layers)
        flatten                 = layers.Flatten()(saved_model.layers[length//2+1].output)
        x                       = layers.Dense(dense_num,activation="relu")(flatten)
        out                     = layers.Dense(10, activation="softmax")(x)
        classification_model    = keras.Model(saved_model.input, out)

        print(classification_model.summary())

        for layer_freeze in classification_model.layers[:-3]:
            layer_freeze.trainable = False

        for l in classification_model.layers:
            print("l.name---->", l.name, "l.trainable\t\t---->", l.trainable, "\t\ttrainable_weights---->", len(l.trainable_weights))

        STEPS_PER_EPOCH = int(1e4)//batch_sz
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            0.001,
            decay_steps=STEPS_PER_EPOCH*1000,
            decay_rate=1,
            staircase=False
        )

        # step = np.linspace(0,100000)
        # lr = lr_schedule(step)
        # plt.figure(figsize = (8,6))
        # plt.plot(step/STEPS_PER_EPOCH, lr)
        # plt.ylim([0,max(plt.ylim())])
        # plt.xlabel('Epoch')
        # _ = plt.ylabel('Learning Rate')

        opt = keras.optimizers.Adam(learning_rate=lrate)
        # opt = keras.optimizers.Adam(lr_schedule)
        
        classification_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
        
        callbackmod2=[
            keras.callbacks.EarlyStopping(patience=4),
            callbacks.ModelCheckpoint(filepath='model2_classification.h5',save_best_only=True,monitor='val_loss',mode='min'),
            keras.callbacks.TensorBoard(log_dir="logs2",
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq="epoch",
            profile_batch=2,
            embeddings_freq=0,
            embeddings_metadata=None)
        ]

        history2 = classification_model.fit(
            train_data, onehot_train_labels,
            epochs=Epochs_Num,
            batch_size=batch_sz,
            validation_data=(test_data, onehot_test_labels),
            callbacks=[callbackmod2]
        )

        predicted_labels = classification_model.predict(unknown_X_TEST)

        num_images_to_show = 2
        for im_ind in range(num_images_to_show):
            
            plot_ind = im_ind*2 + 1
            rand_ind = np.random.randint(low=0, high=unknown_X_TEST.shape[0])

            result = np.where(predicted_labels[rand_ind] == np.amax(predicted_labels[rand_ind]))

            plt.figure(figsize=((13,18)))
            strTit = "Prediction number:" + str(result[0][0]) + ",right class" + str(unknown_y_TEST[rand_ind])
            plt.title(strTit)
            plt.plot(num_images_to_show, 2, plot_ind)
            plt.imshow(unknown_X_TEST[rand_ind, :, :], cmap="gray")
            plt.show()


        opt= keras.optimizers.Adam(learning_rate=lrate_2)

        for layer_freeze in classification_model.layers[:-3]:
            layer_freeze.trainable = True

        classification_model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

        callbackmod3=[
            keras.callbacks.EarlyStopping(patience=2),
            callbacks.ModelCheckpoint(filepath='model_classification_full.h5',save_best_only=True,monitor='val_loss',mode='min',),
            keras.callbacks.TensorBoard(log_dir="logs3",
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq="epoch",
            profile_batch=2,
            embeddings_freq=0,
            embeddings_metadata=None)
        ]
        
        history3 = classification_model.fit(
            train_data, onehot_train_labels,
            epochs=Epochs_Num,
            batch_size=batch_sz,
            validation_data=(test_data, onehot_test_labels),
            callbacks=[callbackmod3]
        )

        predicted_labels_2 = classification_model.predict(unknown_X_TEST)
        
        num_images_to_show = 2
        for im_ind in range(num_images_to_show):
            
            plot_ind = im_ind*2 + 1
            rand_ind = np.random.randint(low=0, high=unknown_X_TEST.shape[0])
            
            result = np.where(predicted_labels_2[rand_ind] == np.amax(predicted_labels_2[rand_ind]))

            plt.figure(figsize=((13,18)))
            strTit = "Prediction number:" + str(result[0][0]) + ",right class" + str(unknown_y_TEST[rand_ind])
            plt.title(strTit)
            plt.plot(num_images_to_show, 2, plot_ind)
            plt.imshow(unknown_X_TEST[rand_ind, :, :], cmap="gray")
            plt.show()

        loss        = history2.history['loss']
        val_loss    = history2.history['val_loss']
        acc         = history2.history['categorical_accuracy']
        val_acc     = history2.history['val_categorical_accuracy']
        
        print("Frozen weights")
        print("min(val_loss)", min(val_loss))
        print("Max training accuracy",max(acc))
        print("Max validation accuracy", max(val_acc))

        fig = plt.figure( figsize=((15,20)) )
        ax1 = fig.add_subplot(221)
        ax1.title.set_text('Loss and Val Loss')
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('loss')
        ax1.plot(loss)
        ax1.plot(val_loss)
        ax1.legend( ['loss', 'val_loss'] )
        
        ax2 = fig.add_subplot(222)
        ax2.title.set_text('Accuracy and Accuracy Loss')
        ax2.set_xlabel('epochs')
        ax2.set_ylabel('accuracy')
        ax2.plot(acc)
        ax2.plot(val_acc)
        ax2.legend( ['accuracy', 'val_accuracy'] )
        plt.show()

        loss        = history3.history['loss']
        val_loss    = history3.history['val_loss']
        acc         = history3.history['categorical_accuracy']
        val_acc     = history3.history['val_categorical_accuracy']
        
        print("Unfrozen weights")
        print("min(val_loss)", min(val_loss))
        print("Max training accuracy",max(acc))
        print("Max validation accuracy", max(val_acc))
        
        fig = plt.figure( figsize=((15,20)) )
        ax1 = fig.add_subplot(221)
        ax1.title.set_text('Loss and Val Loss')
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('loss')
        ax1.plot(loss)
        ax1.plot(val_loss)
        ax1.legend( ['loss', 'val_loss'] )
        
        ax2 = fig.add_subplot(222)
        ax2.title.set_text('Accuracy and Accuracy Loss')
        ax2.set_xlabel('epochs')
        ax2.set_ylabel('accuracy')
        ax2.plot(acc)
        ax2.plot(val_acc)
        ax2.legend( ['accuracy', 'val_accuracy'] )
        plt.show()