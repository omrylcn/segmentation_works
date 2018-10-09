
import keras 
from keras import layers
from keras import Sequential


def basic_segnet(n_classes=10,input_height=400,input_width=480,kernel=3,filter_size=[64,128,256,512],pad=1,pool_size=2):
    
    model=Sequential()
    model.add(layers.Layer(input_shape=(input_height,input_width,3)))

    #encoder
    model.add(layers.ZeroPadding2D(padding=(pad,pad)))
    model.add(layers.Conv2D(filters=filter_size[0],kernel_size=(kernel,kernel),padding="valid"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPool2D(pool_size=(pool_size,pool_size)))

    model.add(layers.ZeroPadding2D(padding=(pad,pad)))
    model.add(layers.Conv2D(filters=filter_size[1],kernel_size=(kernel,kernel),padding="valid"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPool2D(pool_size=(pool_size,pool_size)))


    model.add(layers.ZeroPadding2D(padding=(pad,pad)))
    model.add(layers.Conv2D(filters=filter_size[2],kernel_size=(kernel,kernel),padding="valid"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPool2D(pool_size=(pool_size,pool_size)))

    model.add(layers.ZeroPadding2D(padding=(pad,pad)))
    model.add(layers.Conv2D(filters=filter_size[3],kernel_size=(kernel,kernel),padding="valid"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))

    #decoder

    model.add(layers.ZeroPadding2D(padding=(pad,pad)))
    model.add(layers.Conv2D(filter_size[3],kernel_size=(kernel,kernel),padding="valid"))
    model.add(layers.BatchNormalization())

    model.add(layers.UpSampling2D(size=(pool_size,pool_size)))
    model.add(layers.ZeroPadding2D(padding=(pad,pad)))
    model.add(layers.Conv2D(filter_size[2],kernel_size=(kernel,kernel),padding="valid"))
    model.add(layers.BatchNormalization())

    model.add(layers.UpSampling2D(size=(pad,pad)))
    model.add(layers.ZeroPadding2D(padding=(pad,pad)))
    model.add(layers.Conv2D(filter_size[1],kernel_size=(kernel,kernel),padding="valid"))
    model.add(layers.BatchNormalization())

    model.add(layers.UpSampling2D(size=(pad,pad)))
    model.add(layers.ZeroPadding2D(padding=(pad,pad)))
    model.add(layers.Conv2D(filter_size[0],kernel_size=(kernel,kernel),padding="valid"))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(n_classes,1,padding="valid"))
    model.add(layers.Reshape((model.output_shape[-2] * model.output_shape[-3],n_classes),
                      input_shape=(model.output_shape[-3], model.output_shape[-2],n_classes)))


    model.add(layers.Activation("softmax"))
    return model

