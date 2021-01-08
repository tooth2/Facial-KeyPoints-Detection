# Facial-KeyPoints-Detection
Applied image processing techniques and deep learning techniques to detect faces in an image and find facial keypoints. Facial keypoints include points around the eyes, nose, and mouth on a face and are used in many applications. These applications include: facial tracking, facial pose recognition, facial filters, emotion recognition etc.

## Objective
The objective is to predict keypoint positions on face images. This can be used as a building block in several applications such as tracking faces in images and videos, analysing facial expressions, biometrics / face recognition and so on.
This project is well known as [Kaggle challenge](https://www.kaggle.com/c/facial-keypoints-detection)
> "Detecing facial keypoints is a very challenging problem.  Facial features vary greatly from one individual to another, and even for a single individual, there is a large amount of variation due to 3D pose, size, position, viewing angle, and illumination conditions. Computer vision research has come a long way in addressing these difficulties, but there remain many opportunities for improvement."
<img src = "images/key_pts_example.png" width = "300" />

## Project Goal 
The project goal is to combine computer vision techniques and deep learning architectures to build a facial keypoint detection system that takes in any image with faces, and predicts the location of 68 distinguishing key points on each face. 
* image processing and feature extraction techniques to programmatically represent different facial features
* deep learning techniques to program a convolutional neural network to recognize facial keypoints

## Implementation Approach
### DATA 
Facial keypoints (also called facial landmarks) are represented by 68 keypoints in a single face, with coordinates (x, y), for that face. These keypoints mark important areas of the face: the eyes, corners of the mouth, the nose, like the image below etc. These keypoints are relevant for a variety of tasks, such as face filters, emotion recognition, pose recognition, and so on. Here they are, numbered, are arranged and matched to represent different portions of the face.
<img src="images/landmarks_numbered.jpg" width = "300"/>
The original set of face image data was extracted from the YouTube Faces Dataset, and the facial keypoints dataset consists of 5770 color images. All of these images are summarized in CSV files which include keypoint's (x,y) coordinates
* training data set: 3462 images' keypoints
* testing data set: 2308 images' keypoints

