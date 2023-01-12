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
        print("magic_number",       magic_number)
        print("number_images",      number_images)
        print("number_of_rows",     number_of_rows)
        print("number_of_columns",  number_of_columns)
        f.close()
        return (npimages, magic_number, number_images, number_of_rows, number_of_columns)

def Convpool(convolutionx, numfilters, convsize, convNums):
  convolutionret = convolutionx
  for i in range(0,convNums):
    convolutionret = layers.Conv2D(numfilters[i],convsize[i],activation='relu',strides=1,padding='same')(convolutionret)
    convolutionret = layers.BatchNormalization()(convolutionret)
  poolingret = layers.MaxPooling2D(pool_size=(2,2))(convolutionret)
  return poolingret

def Convup(convolutionx, numfilters, convsize, convNums):
  convolutionret = convolutionx
  for i in range(convNums-1,-1,-1):
    convolutionret = layers.Conv2D(numfilters[i],convsize[i],activation='relu',strides=1,padding='same')(convolutionret)
    convolutionret = layers.BatchNormalization()(convolutionret)
  upsamplingret = layers.UpSampling2D((2,2))(convolutionret)
  return upsamplingret

def encoder(imageinput, BlocksNum, FiltersNum, FilterSize, Convolutions_per_Block):
  blockreturn = imageinput
  for i in range(0,BlocksNum):
    blockreturn = Convpool(blockreturn,FiltersNum[i],FilterSize[i],Convolutions_per_Block[i])
  for i in range(0,Convolutions_per_Block[-1]):
    blockreturn = layers.Conv2D(FiltersNum[-1][i],FilterSize[-1][i],activation='relu',padding = 'same')(blockreturn)
    blockreturn = layers.BatchNormalization()(blockreturn)
  return blockreturn

def decoder(convolution3, BlocksNum, FiltersNum, FilterSize, Convolutions_per_Block):
  blockreturn = convolution3
  for i in range(Convolutions_per_Block[-1]-1,-1,-1):
    blockreturn = layers.Conv2D(FiltersNum[-1][i],FilterSize[-1][i],activation='relu',padding = 'same')(blockreturn)
    blockreturn = layers.BatchNormalization()(blockreturn)
  for i in range(BlocksNum-1,-1,-1):
    blockreturn = Convup(blockreturn,FiltersNum[i],FilterSize[i],Convolutions_per_Block[i])
  decoded = layers.Conv2D(1,FilterSize[0][0],activation='sigmoid',padding='same')(blockreturn) #28 x 28 x 1
  return decoded

if __name__ == "__main__":
  # python autoencoder.py -d ./t10k-images.idx3-ubyte
  learn_rate = 0.0001
  batchSize   = 128
  Epochs_Num  = 100
  Block_Num   = 2 # we define a block as a combination of Convolution,Batch Normalization and pooling - Restriction, max = 3
  Convolutions_per_Block = (2, 2, 3) # we may make more convolutions than one per block / we include last block that may not enclude pooling
  FilterSize =  ( ((3,3),(3,3)), ((3,3),(3,3)), ((3,3),(3,3),(3,3)) ) # filter size for each Convolution
  FiltersNum = ( (32,32), (64,64), (128,128,256) ) # number of filters per Convolution

  with tf.device("gpu:0"):

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--Dataset", help="Insert dataset file", required="yes")
    args = parser.parse_args()
    FileDataSet = args.Dataset
    print( "\nDataset {}".format(FileDataSet))
    array_images, _, nImages, _, _ = parse_file(FileDataSet)
    
    print("array_images.shape", array_images.shape)
    
    identities = np.arange(nImages) # no use
    X_train, X_test, _, _ = train_test_split(array_images, identities, test_size=0.3, shuffle=True, random_state=11)
    X_train = X_train / 255
    X_test  = X_test / 255
    
    print("X_test.shape", X_test.shape)
    print("X_train.shape", X_train.shape)

    ##build NN##
    input_img = keras.Input( shape=(28,28,1) )

    STEPS_PER_EPOCH = int(1e4)//batchSize
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

    convolution3    = encoder( input_img,     Block_Num, FiltersNum, FilterSize, Convolutions_per_Block )
    decoded         = decoder( convolution3,  Block_Num, FiltersNum, FilterSize, Convolutions_per_Block )
    autoencoder     = keras.Model( input_img, decoded )
    opt             = keras.optimizers.Adam( learning_rate=learn_rate )
    # opt             = keras.optimizers.Adam(lr_schedule)

    autoencoder.compile( optimizer=opt, loss='mean_squared_error', metrics=['accuracy', 'mse'] )

    callbackmod=[
      keras.callbacks.EarlyStopping(patience=4),
      callbacks.ModelCheckpoint( filepath='model.h5', save_best_only=True, monitor='val_loss', mode='min' ),
      keras.callbacks.TensorBoard(
        log_dir="logs",
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq="epoch",
        profile_batch=2,
        embeddings_freq=0,
        embeddings_metadata=None
      )
    ]

    history1 = autoencoder.fit(
      X_train, X_train,
      epochs=Epochs_Num,
      batch_size=batchSize,
      validation_data=( X_test, X_test ),
      callbacks=[callbackmod]
    )

    print(autoencoder.summary())

    decoded_imgs        = autoencoder.predict(X_test)
    decoded_images_orig = np.reshape( decoded_imgs, newshape=(decoded_imgs.shape[0], 28, 28) )

    num_images_to_show = 3
    for im_ind in range(num_images_to_show):
      plot_ind = im_ind * 2 + 1
      rand_ind = np.random.randint(low=0, high=X_test.shape[0])
      
      fig = plt.figure( figsize=((13,18)) )
      ax1 = fig.add_subplot(221)
      ax1.title.set_text('Actual Image')
      ax1.plot(num_images_to_show, 2, plot_ind)
      ax1.imshow(X_test[rand_ind, :, :], cmap="gray")
      
      ax2 = fig.add_subplot(222)
      ax2.title.set_text('Encoded Image')
      ax2.plot(num_images_to_show, 2, plot_ind+1)
      ax2.imshow(decoded_images_orig[rand_ind, :, :], cmap="gray")
      plt.show()

    loss        = history1.history['loss']
    val_loss    = history1.history['val_loss']
    acc         = history1.history['accuracy']
    val_acc     = history1.history['val_accuracy']

    print("min(val_loss)", min(val_loss))
    print("max(val_acc)", max(val_acc))

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