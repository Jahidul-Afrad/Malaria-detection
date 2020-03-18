# Malaria-detection
Malaria is a serious disease caused by a peripheral blood parasite of the genus Plasmodium, 
transmitted to humans via mosquitoes. Malaria is endemic in 13 of 64 districts in Bangladesh. 
The three hilly districts (Rangamati, Khagrachari and Banderban) account for 80% of the total burden of malaria.


# Introduction


In this paper we propose a detection of malaria from red blood cell based on Convolutional neural network. 
First, the original image is resize its shape and rotate them 30, 60, 90 degree and then using a convolutional 
neural network as a pixel classifier to localize the erythrocytes. In this work, we have used deep convolution neural 
networks (CNN) with architecture of VGG19. The pre-trained VGG19 architecture is retrained with 81270 preprocessed and 
leveled images. The dataset is preprocessed using multiple high performing and effective image processing techniques. 
Then the new trained models are used for identifying the malarial blood cell. We have achieved 96.724% accuracy in the 
classification. This report represent computer vision and image analysis studies aiming at automated diagnosis or screening
of malaria infection in microscope images of thin blood film smears based on the use of image processing and neural network.


# Motivation 
Malaria is a public health problem in 90 countries around the world, affecting 300 million people and responsible directly for about one million deaths annually. Bangladesh is about 150 million; 14 million people live in the 13 malaria-endemic districts. During 2009 a total of 63,873 cases and 47 deaths were reported. The malaria parasite enters the bloodstream, multiplies in the liver cells, and is then released back into the bloodstream, where it infects and destroys red blood cells. Its diagnosis is usually done manually by compound light microscopy which is time consuming. Early detection can make an effective treatment. An automated system can play a vital role to this state.
# Objectives and Specific Aims
	To preprocess the data
	To convert the image into numeric value
	Improved accuracy and implement a new approach 
	Computer Aided Diagnose (CAD) system
	Detect malaria in an earlier stage
	Give treatment in a short period of time
	Reduce hard manual labor
	Finding better solution for malaria detection



# Proposed method


# A. Cropping image 
The original image contains black and dark portion in background. Hence, we crop the image keeping an aspect ratio of 3:2 of 
original image. 

# B. Resizing image 
The images have higher resolution as an example an image is 127× 154. We have downscaled the images to 64×64 pixels. 
Resized image of 64×64 pixels.

# C. Rotation (30, 60 and 90) 

In technique 1, we have rotate the green channel from the RGB color image. Then apply 30 degree rotation because for this
we find the infected pixel on 30 degree and for this we can increase our dataset image. Similarly we rotate the image on 
60 degree and 90 degree. By this we can increase our dataset on 27160 * 3 = 81480. 

# D. Leveling data
 
Before training and testing, we leveled the image of data by infected and uninfected.


# Visual Geometry Group (VGG19)


In our experiment we have used VGG19 architecture as the deep convolution neural network. VGG19 is trained for the ImageNet
Large Visual Recognition Challenge using the data from 2014. This is a standard task in computer vision, where models try to 
extract feature form entire images. By now we would’ve already noticed that CNNs were starting to get deeper and deeper. 
This is because the most straightforward way of improving performance of deep neural networks is by increasing their size. 
The folks at Visual Geometry Group (VGG) invented the VGG-19 which has 17 convolutional and 3 fully-connected layers, 
carrying with them the ReLU tradition from AlexNet. This network stacks more layers onto AlexNet, and use smaller size filters
(1×1 and 3×3). It consists of 138M parameters and takes up about 549MB of storage space. VGG19 is composed of blocks of 3x3 filters
separated by max-pooling layers. We can visualize what features each filter captures by learning the input image that maximizes
the activation of that filter. The input image is initially random while the loss is calculated as the activation of a 
particular filter. Using gradient ascent to maximize this loss generates synthetic images that capture what a filter learns.

To compare models, it is required to examine how often the model fails to predict the correct answer. If the input images are greater than 128 x 128, its kernel size may choose to use greater than 3 (kernel size>3) because of learn larger spatial filter and to help reduce volume size where the VGG19 exclusively use kernel size (3 x 3). For that the model fails to predict the correct answer.
As our image of dataset are 64 x 64, for that we choose kernel size<3 which is not good for advance CNN architecture such as inceptionV3, ResNet50 etc. Because they use kernel 1x1, 3x3, 5x5 and so on. And combine the output. Again we choose strides 2x2 because our dataset of image size is small. The final dense layer softmax for a particular category. Gradient ascent is applied until the loss reaches _ 1 which implies that the CNN is 100% confident that the synthetic image is that category


# Accuracy
We have considered Precession, Recall, F1 score and accuracy which is given below:
Precession:97.96%
Recall:94.69%
F1 score:96.36%
Accuracy:96.78%


# Concluding Remarks
With a limited number of medical staff, an automated system can significantly decrease the tedious manual labor involved in diagnosing large quantities of malaria blood images. Image processing based diagnosis has played a dominant role in previous studies. However, the progress made in deep convolutional neural networks and features extraction has led to them becoming a state-of-the-art technique in blood image for detect malaria.
