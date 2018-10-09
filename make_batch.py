
import numpy as np
import cv2
import glob
import itertools
import os
import matplotlib.pyplot as plt



def getImageArr(path,width,height,imgNorm="sub_mean"):
    
    try:
        
        img=cv2.imread(path,1)
        
        
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
            img=img/255
        

        
        
        return img
    
    except:
        print("get image Error")
        


def getSegmentationArr( path , nClasses ,  width , height  ):

    seg_labels = np.zeros((  height , width  , nClasses ))
    try:
        img = cv2.imread(path, 1)
        img = cv2.resize(img, ( width , height ))
        img = img[:, : , 0]

        for c in range(nClasses):
            seg_labels[: , : , c ] = (img == c ).astype(int)

    except:
        print("error")

    seg_labels = np.reshape(seg_labels, ( width*height , nClasses ))
    return seg_labels


def imageSegmentationGemerator(images_path,segs_path,batch_size,n_classes,
                              input_h,input_w,output_h,output_w):
    
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
    
    while True:
        x=[]
        y=[]
        for _ in range(batch_size):
            im,seg=next(zipped)
            de=getImageArr(im,input_w,input_h)
           
            x.append(de)
            y.append(getSegmentationArr(seg,n_classes,output_w,output_h))
            
        yield np.array(x), np.array(y) 
        

