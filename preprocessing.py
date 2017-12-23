import cv2
import os
import numpy as np
import copy
from shutil import copyfile
from PIL import Image
from PIL import ImageFilter

idx = 0
i = 10

#Load Images from a Folder containing images
def load_images_from_folder(path):

    valid_images = [".jpg",".gif",".png",".tga"]
    images = []
    names = []
    for f in os.listdir(path):
        ext = os.path.splitext(f)
        if ext[1].lower() in valid_images:
            names.append(ext)
            images.append(cv2.imread(os.path.join(path,f),cv2.IMREAD_GRAYSCALE))

    print(len(images))
    return images,names

#Copy Files from one folder to another after Preprocessing
def copy_files(path,path1):

    files = os.listdir(path)
    for file in files:
        copyfile(path + '/' + file,path1 +'/' + file)            

#Apply Filters to each image in the path 
def filter_segment(path):
    valid_images = [".jpg", ".gif", ".png", ".tga"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)
        if ext[1].lower() in valid_images:
            images = Image.open(os.path.join(path,f))
            images = images.filter(ImageFilter.UnsharpMask(radius = 0,percent = 200,threshold = 7))
        im = np.array(images)
        cv2.imwrite(path + "/" + ext[0] + ".jpg", im)             

#Segmentation of Images
def segment(image):
    Kernel = np.ones((5,5),np.uint8)
    im = copy.copy(image)
    ret,thresh = cv2.threshold(im,127,255,0)
    thresh = cv2.bitwise_not(thresh)
    im2,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(im2,contours,-1,(0,255,0),3)

    lstBb = []
    for cnt in contours:
        lstBb.append(cv2.boundingRect(cnt))

    cntbb = [(c,b) for (c,b) in zip(contours,lstBb)]
    cntbb = sorted(cntbb, key=lambda x: x[1][0])
    cntbb = sorted(cntbb, key=lambda x: x[1][1])
    cnts = []
    cnts = [c for c,b in cntbb]
    bb = np.array([b for c,b in cntbb])

    f = open("../OCR/document.txt","w+")
    global idx
    for x,y,w,h in bb:
        roi = image[y-3 : y+h+3, x-3 : x+w+3]
        f.write("%d %d %d %d\n" %(x+(w/2),y+(h/2),w,h))
        cv2.rectangle(im,(x-3,y-3),(x+w+3,y+h+3),(0,255,0),2)
        if roi.any():
            cv2.imwrite("../OCR/segmented/" + str(idx) + ".jpg",roi)       
        idx+=1
    f.close()
    global i
    cv2.imwrite("../OCR/segmented/contours/" + str(i) + '.jpg',im)
    i+=1

#Grayscaling the image
def grayscaling(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return gray

#OTSU's Thresholding 
def thresholding(gray):
    thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    return thresh

#Program for Deskewing the Image if it exists
def deskew(image):
    gray = cv2.bitwise_not(image)
    thresh = thresholding(gray)[1]
    coords = np.column_stack(np.where(thresh>0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle<-45:
        angle = -(90+angle)
    else:
        angle = -angle
    (h,w) = image.shape[:2]
    center = (w//2,h//2)
    M = cv2.getRotationMatrix2D(center,angle,1.0)
    rotated = cv2.warpAffine(image,M,(w,h),flags = cv2.INTER_CUBIC,borderMode = cv2.BORDER_REPLICATE)
    return rotated

if __name__ == '__main__':
    
    images = []

    #input = Folder containing images
    print("Reading Documents.........\n\n")
    images,names = load_images_from_folder("../OCR/input")  

    l = len(images)
    print("PreProcessing and Segmentation")

    for i in range(l):
        images[i] = deskew(images[i])
        cv2.imwrite("../OCR/preprocessed/prepros.jpg" ,images[i])            
        segment(images[i])               
    print("Applying Filters for contours")  
    filter_segment("../OCR/segmented")                        
    copy_files('../OCR/segmented','../OCR/input_contours')                      ########################## add path
    print("Preprocessing DONE.................")





