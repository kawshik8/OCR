# OCR
Object Character Recognition

I created an OCR (Object Character Recognition) Engine during my internship at Access Health Care Limited, Chennai, Tamil Nadu, India. The programs are in python and have implemented using computer vision libraries like OpenCv and PIL (Pillow now). The entire workflow was created from scratch which included problems like finding features, drawing contours (bounding boxes), morphological transformations, smoothing and sharpening images, finding and fixing skews in images and finally segmentation of images using contours. An SVM classifier was then used to detect various classes in the contours. Non max suppression was then used to prevent overlapping of contours in the images. This prediction was then used to form a complete workflow to create an OCR Engine. 
