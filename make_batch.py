import numpy as np
import cv2
import glob
import itertools
import os
import matplotlib.pyplot as plt


def getImageArr(img,width,height,imgNorm="divide"):
    
    try:
                
        if imgNorm=="sub_and_divide":
            img = np.float32(cv2.resize(img, ( width , height ))) / 127.5 - 1
        
        elif imgNorm=="sub_mean":
            img=cv2.resize(img,(width,height))
            img=img.astype(np.float32)
            img[:,:,0]-=103.939
            img[:,:,1]-=116.779
            img[:,:,2]-=123.68
        
        elif imgNorm=="divide":
            img=cv2.resize(img,(width,height))
            img=img.astype(np.float32)
            img=img*1./255
         
        return img
    
    except:
        print("Error")
        


def aug_data(image=None,segmentation=None,train=True):
    
    if train:
        
        i1=np.random.choice([-1,1])
        i2=np.random.choice([-1,1])
        k1=np.random.randint(0,20)
        k2=np.random.randint(0,20)

        tim=image[::i1,::i2]
        tseg=segmentation[::i1,::i2]

        tim=tim[k1*10:(100+k1*30),k2:(100+k2*k1)]
        tseg=tseg[k1*10:(100+k1*30),k2:(100+k2*k1)]
        return tim,tseg

    else:
        
        return image,segmentation



def getSegmentationArr(img , nClasses ,  width , height  ):

    seg_labels = np.zeros((  height , width  , nClasses ))
    try:
        
        img = cv2.resize(img, ( width , height ))
        img = img[:, : , 0]

        for c in range(nClasses):
            seg_labels[: , : , c ] = (img == c ).astype(int)

    except:
        print("error")

    seg_labels = np.reshape(seg_labels, ( width*height , nClasses ))

    return seg_labels




def imageSegmentationGenerator(images_path,segs_path,batch_size,n_classes,
                              input_h,input_w,output_h,output_w,train=True):
    
    assert images_path[-1]=="/"  
    assert segs_path[-1]=="/"
    
    images = glob.glob( images_path + "*.jpg"  ) + glob.glob( images_path + "*.png"  ) +  glob.glob( images_path + "*.jpeg"  )
    images.sort()
    segmentations=glob.glob( segs_path + "*.jpg"  ) + glob.glob( segs_path + "*.png"  ) +  glob.glob( segs_path + "*.jpeg"  )
    segmentations.sort()
    
    assert len(images)==len(segmentations)
    for im,seg in zip(images,segmentations):
        assert (im.split("/")[-1].split(".")[0]==seg.split("/")[-1].split(".")[0] )
    zipped=itertools.cycle(zip(images,segmentations))
    tr=train
    while True:
        x=[]
        y=[]
        for _ in range(batch_size):
            im_path,seg_path=next(zipped)
            im=cv2.imread(im_path)
            seg=cv2.imread(seg_path)
            im,seg=aug_data(im,seg,train=tr)
        
            de=getImageArr(im,input_w,input_h)
            x.append(de)
            y.append(getSegmentationArr(seg,n_classes,output_w,output_h))
        
        yield np.array(x), np.array(y)
        


if __name__ == "__main__":
    
    path="data/dataset/"
    a=os.listdir(path)
    tr=os.path.join(path,a[2]+"/")
    an=os.path.join(path,a[3]+"/")
    gen=imageSegmentationGenerator(tr,an,2,12,300,400,200,300,train=True)
    res=next(gen)
    plt.imshow(res[0][1])
    plt.show()






