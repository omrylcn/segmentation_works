from keras.models import *
from keras.layers import *
from keras.regularizers import l2
import os 



path="vgg16_weights_tf_dim_ordering_tf_kernels.h5"



def get_model_weights():
    if os.path.exists(path):
        print("model weights are exist")
    else :
        get_ipython().system('wget "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5"')
        print("model weights are downloaded")



def crop( o1 , o2 , i  ):
    o_shape2 = Model( i  , o2 ).output_shape
    outputHeight2 = o_shape2[1]
    outputWidth2 = o_shape2[2]

    o_shape1 = Model( i  , o1 ).output_shape
    outputHeight1 = o_shape1[1]
    outputWidth1 = o_shape1[2]
   
    cx = abs( outputWidth1 - outputWidth2 )
    cy = abs( outputHeight2 - outputHeight1 )

    if outputWidth1 > outputWidth2:
        o1 = Cropping2D( cropping=((0,0) ,  (  0 , cx )))(o1)
        
    else:
        o2 = Cropping2D( cropping=((0,0) ,  (  0 , cx )))(o2)
           
    if outputHeight1 > outputHeight2 :
        o1 = Cropping2D( cropping=((0,cy) ,  (  0 , 0 )))(o1)
    else:
        o2 = Cropping2D( cropping=((0, cy ) ,  (  0 , 0 )) )(o2)

    return o1 , o2



def FCN8(    
    n_classes=12,
    input_height=416,
    input_width=608,
    vgg_level=3,
    weight_decay=0.,
    batch_momentum=0.9,
    batch_shape=None,
    pretrained=True):

    img_input=Input(shape=(input_height,input_width,3))

    x=Conv2D(64,(3,3),activation="relu",padding="same",name="block1_conv1")(img_input)
    x=Conv2D(64,(3,3),activation="relu",padding="same",name="block1_conv2")(x)
    x=MaxPool2D((2,2),strides=(2,2),name="block1_pool")(x)
    f1=x
    #Block2
    x=Conv2D(128,(3,3),activation="relu",padding="same",name="block2_conv1")(x)
    x=Conv2D(128,(3,3),activation="relu",padding="same",name="block2_conv2")(x)
    x=MaxPool2D((2,2),strides=(2,2),name="block2_pool")(x)
    f2=x
    #Block3
    x=Conv2D(256,(3,3),activation="relu",padding="same",name="block3_conv1")(x)
    x=Conv2D(256,(3,3),activation="relu",padding="same",name="block3_conv2")(x)
    x=Conv2D(256,(3,3),activation="relu",padding="same",name="block3_conv3")(x)
    x=MaxPool2D((2,2),strides=(2,2),name="block3_pool")(x)
    f3=x
    #Block4
    x=Conv2D(512,(3,3),activation="relu",padding="same",name="block4_conv1")(x)
    x=Conv2D(512,(3,3),activation="relu",padding="same",name="block4_conv2")(x)
    x=Conv2D(512,(3,3),activation="relu",padding="same",name="block4_conv3")(x)
    x=MaxPool2D((2,2),strides=(2,2),name="block4_pool")(x)
    f4=x
    #Block5
    x=Conv2D(512,(3,3),activation="relu",padding="same",name="block5_conv1")(x)
    x=Conv2D(512,(3,3),activation="relu",padding="same",name="block5_conv2")(x)
    x=Conv2D(512,(3,3),activation="relu",padding="same",name="block5_conv3")(x)
    x=MaxPool2D((2,2),strides=(2,2),name="block5_pool")(x)
    f5=x


    #vgg=Model(img_input,x)
    #vgg.load_weight(path,by_name=True)

    o=f5


    #o= Conv2D( 4096 , ( 7 , 7 ) , activation='relu' , padding='same')(o)
    #o = Dropout(0.5)(o)
    #o = Conv2D( 4096 , ( 1 , 1 ) , activation='relu' , padding='same')(o)
    #o = Dropout(0.5)(o)

    o=Conv2D(4096,(1,1),activation="relu",padding="same",kernel_regularizer=l2(weight_decay))(o)
    o=DepthwiseConv2D((7,7),activation="relu",padding="same",kernel_regularizer=l2(weight_decay))(o)
    o=Dropout(0.5)(o)
    o=DepthwiseConv2D((1,1),activation="relu",padding="same",kernel_regularizer=l2(weight_decay))(o)
    o=Dropout(0.5)(o)


    """
    -=:{ Conv2DTranspose output shape}:=- 

    new_rows = ((rows - 1) * strides[0] + kernel_size[0]
                - 2 * padding[0] + output_padding[0])
    new_cols = ((cols - 1) * strides[1] + kernel_size[1]
                - 2 * padding[1] + output_padding[1])

    """

    o=Conv2D(n_classes,(1,1),kernel_initializer="he_normal")(o)
    o=Conv2DTranspose(n_classes,kernel_size=(4,4),strides=(2,2),use_bias=False)(o)

    # o and o2 shapes are not same. That is problem to add operation.Therefore we use crop operation

    o2=f4
    o2=Conv2D(n_classes,(1,1),kernel_initializer="he_normal")(o2)
    o,o2=crop(o,o2,img_input)
    o=Add()([o,o2])    
    o=Conv2DTranspose(n_classes,kernel_size=(4,4) ,  strides=(2,2) , use_bias=False)(o)


    o2 = f3 
    o2 = Conv2D(n_classes,( 1 , 1 ) ,kernel_initializer='he_normal' )(o2)
    o2,o = crop( o2 , o , img_input )
    o = Add()([ o2 , o ])
    o = Conv2DTranspose( n_classes , kernel_size=(16,16) ,  strides=(8,8) , use_bias=False)(o)

    global o_shape
    o_shape=Model(img_input,o).

    o=Reshape((o_shape[1]*o_shape[2],n_classes))(o)
    o=Activation("softmax")(o)

    model=Model(img_input,o)

    model.outputs=o_shape
    
    if pretrained:
        get_model_weights()
        model.load_weights(path,by_name=True)
    
    return model



