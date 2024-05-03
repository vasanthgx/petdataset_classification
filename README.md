

![logo](https://github.com/vasanthgx/petdataset_classification/blob/main/images/logo.gif)


# Project Title


**Image Segmentaion with Breed Classification ML Project**
 <img src="https://github.com/Anmol-Baranwal/Cool-GIFs-For-GitHub/assets/74038190/b3fef2db-e671-4610-bb84-1d65533dc5fb" width="300" align='right'>

<br><br>






## Introduction

This project aims to demonstrate the application of machine learning techniques in two key areas of computer vision: image segmentation and breed classification. Image segmentation involves partitioning an image into multiple segments or regions to simplify its representation and facilitate further analysis. Breed classification, on the other hand, focuses on identifying the breed of animals depicted in images, in our case, dogs.


## Project Overview

In this project, we have developed a machine learning pipeline that combines state-of-the-art techniques for image segmentation and breed classification. The pipeline processes input images of dogs, performs semantic segmentation to identify distinct regions within the images, and subsequently classifies the breed of the dog present in each segmented region.

## Key Features

- **Image Segmentation**: We utilize advanced convolutional neural network (CNN) architectures to perform semantic segmentation, enabling precise delineation of different objects or regions within the input images.
- **Breed Classification**: Leveraging transfer learning, we fine-tune pre-trained CNN models to classify the breed of dogs present in the segmented regions. This allows us to achieve high accuracy even with limited training data.
- **End-to-End Pipeline**: Our project provides a seamless end-to-end solution for image segmentation and breed classification, enabling users to input raw images and obtain detailed segmentation masks along with breed predictions.
- **Model Deployment**: [We have deployed the project in  Huggingface / Spaces through Gradio Application.](https://huggingface.co/spaces/Vasanthgx/oxford_pets_breed_classification)

![alt text](https://github.com/vasanthgx/petdataset_classification/blob/main/images/gradio.png)

### Implementation Details

- Dataset: The Oxford-IIIT Pet Dataset (view below for more details)
- Model: [MobileNetV2]('https://keras.io/api/applications/mobilenet/')
- Input: 37 category pet dataset with roughly 200 images for each class
- Output: Segmentation mask , Breed classification

### Dataset Details

[This dataset was obtained from this repository](https://www.robots.ox.ac.uk/~vgg/data/pets/)

The Oxford-IIIT Pet Dataset is a widely used collection of images containing various breeds of cats and dogs. It was created by the Visual Geometry Group at the University of Oxford and the IIIT Delhi. The dataset consists of over 7,000 images of pets belonging to 37 different categories, with each category representing a specific breed.
Each image in the dataset is annotated with bounding boxes and class labels, making it suitable for tasks such as object detection and classification. The dataset provides a diverse range of poses, backgrounds, and lighting conditions, making it valuable for training and evaluating computer vision algorithms.
Researchers and practitioners often use the Oxford-IIIT Pet Dataset for tasks such as fine-grained classification, instance segmentation, and pose estimation. Its availability and richness make it a benchmark dataset in the field of computer vision, contributing to advancements in pet recognition technology and beyond.

### Dog and Cat Breeds in the dataset

 ![alt text](https://github.com/vasanthgx/petdataset_classification/blob/main/images/dataset_stats.png)

### Annotation Examples from the dataset

 ![alt text](https://github.com/vasanthgx/petdataset_classification/blob/main/images/annotation_examples.png)


## Evaluation and Results

### Building a Classification model with Scikit Learn

- Downloading and unzipping the dataset
```
!wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
!wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz

!tar zxvf images.tar.gz
!tar zxvf annotations.tar.gz

```
- viewing an image file from the downloaded images folder
```
from IPython.display import Image
Image('/content/images/Abyssinian_1.jpg')

```
 ![alt text](https://github.com/vasanthgx/petdataset_classification/blob/main/images/sample_cat.jpg)

- Checking the size of the images and resizing them to a standard size

- converting the images and the labels to a numpy array


- Splitting the data into training and test datasets and training a classifier.




### Creating a Segmentation Mask using the Skimage library

- Let us first segment the images with Otsu Thresholding ( details in the FAQ section )

![alt text](https://github.com/vasanthgx/petdataset_classification/blob/main/images/segmask-1.png)

- We illustrate how to apply one of these thresholding algorithms. Otsu’s method calculates an “optimal” threshold (marked by a red line in the histogram below) by maximizing the variance between two classes of pixels, which are separated by the threshold. Equivalently, this threshold minimizes the intra-class variance.

![alt text](https://github.com/vasanthgx/petdataset_classification/blob/main/images/hist.gif)

![alt text](https://github.com/vasanthgx/petdataset_classification/blob/main/images/otsu-2.png)

- If you are not familiar with the details of the different algorithms and the underlying assumptions, it is often difficult to know which algorithm will give the best results. Therefore, Scikit-image includes a function to evaluate thresholding algorithms provided by the library. At a glance, you can select the best algorithm for your data without a deep understanding of their mechanisms.

![alt text](https://github.com/vasanthgx/petdataset_classification/blob/main/images/motsu-1.png)
![alt text](https://github.com/vasanthgx/petdataset_classification/blob/main/images/motsu-2.png)

- [Multi Otsu Thresholding](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_multiotsu.html#sphx-glr-auto-examples-segmentation-plot-multiotsu-py) The multi-Otsu threshold is a thresholding algorithm that is used to separate the pixels of an input image into several different classes, each one obtained according to the intensity of the gray levels within the image.

Multi-Otsu calculates several thresholds, determined by the number of desired classes. The default number of classes is 3: for obtaining three classes, the algorithm returns two threshold values. They are represented by a red line in the histogram below.

![alt text](https://github.com/vasanthgx/petdataset_classification/blob/main/images/motsu-3.png)



### Object Segmentation using Convolutional Neural Networks using [TensorFlow Datasets](https://www.tensorflow.org/datasets/overview)

 **What is Image Segmentaion?( more details in FAQ section)**
In an image classification task, the network assigns a label (or class) to each input image. However, suppose you want to know the shape of that object, which pixel belongs to which object, etc. In this case, you need to assign a class to each pixel of the image—this task is known as segmentation. A segmentation model returns much more detailed information about the image. Image segmentation has many applications in medical imaging, self-driving cars and satellite imaging, just to name a few.



#### [Loading Dataset and Exploration](https://www.tensorflow.org/datasets/catalog/oxford_iiit_pet)


- Exploring sizes and values in dataset
- Displaying the dataset - image along with the Mask and the 3 types of annotated masks
	- Mask Interpretation
		- Object(foreground)
		- Background
		- Ambiguous Region
		
![alt text](https://github.com/vasanthgx/petdataset_classification/blob/main/images/segmask-tf-1.png)

- Pre-processing 

	- Normalize : we use the tf.cast( ) function to convert the images and masks  to float32 type and then divide them by 255.
```

def normalize_img(data):
  """Normalizes images: `uint8` -> `float32`."""
  image = data['image']
  mask = data['segmentation_mask']
  image = tf.image.resize(image, [128, 128])
  mask = tf.image.resize(mask, [128, 128], method='nearest')
  image = tf.cast(image, tf.float32) / 255.0
  mask = tf.cast(mask-1, tf.float32)
  return image, mask

  # 1-1 = 0
  # 2-1 = 1
  # 3-1 = 2
```

#### Model Building

- We first use the MobileNetV2 architecture.
- MobileNetV2 is very similar to the original MobileNet, except that it uses inverted residual blocks with bottlenecking features.It has a drastically lower parameter count than the original MobileNet. MobileNets support any input size greater than 32 x 32, with larger image sizes offering better performance.
	
[This is the link for the reference paper](https://arxiv.org/abs/1801.04381) for the MobileNetV2 pretrained model that we are using for our application
		
[This is link for the reference paper for the MobileNets](https://arxiv.org/abs/1704.04861) - Efficient Convolutional Neural Networks for Mobile vision Applications
		
[Keras provides the MobileNetv2() function](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2) which returns a  Keras image classification model, optionally loaded with weights pre-trained on ImageNet.
		
```
		tf.keras.applications.MobileNetV2(
		input_shape=None,
		alpha=1.0,
		include_top=True,
		weights='imagenet',
		input_tensor=None,
		pooling=None,
		classes=1000,
		classifier_activation='softmax'
	)
		
```
	
- we build a base model using the above MobileNetv2 pretrained model.
- Next we freeze the output of the different layers of the above CNN, to be used later as part the decoding(up-sampling) of the U-Net[see FAQ section for more details ]  architecture
- Next we make use of Pix2Pix model, which is a conditional generative adversarial network (GAN) architecture that learns a mapping from an input image to an output image. We build a upsampling model with this architecture.
- Next we make combine both the base model(down_stack - encoder ) and upsampling model (up_stack - decoder )to build a U-Net architecturej,which is as follows
	
![alt text](https://github.com/vasanthgx/petdataset_classification/blob/main/images/unet.png)
	
- The U-Net architecture is characterized by its symmetric encoder-decoder structure,which enables the network to capture both local and global features while preserving spatial information.
- Building a model using the above U-Net architecture, gives us a model with the following parameters
	
![alt text](https://github.com/vasanthgx/petdataset_classification/blob/main/images/params.png)
	
- We finally fit the model with the training dataset and run it for 10 epochs
	
```
	history = model.fit(
    ds_train,
    epochs=10,
    validation_data=ds_test,
							)
```
	- Comparing the training loss with the validation loss along with the epochs
	
![alt text](https://github.com/vasanthgx/petdataset_classification/blob/main/images/epochs.png)
	

	
#### Evaluation

- We now run the above model on the test dataset  and display the image, segmented mask and the predicted mask
![alt text](https://github.com/vasanthgx/petdataset_classification/blob/main/images/predmask-1.png)
![alt text](https://github.com/vasanthgx/petdataset_classification/blob/main/images/predmask-2.png)




### Breed Classification
- Making use of the fastai and timm libraries. Both are Pytorch libraries.
	- Fastai : The fastai library is an open-source deep learning library built on top of PyTorch, designed to simplify the process of training deep learning models and enable rapid experimentation.
	- Timm :  Short for "PyTorch Image Models," is a collection of pre-trained models for computer vision tasks implemented in PyTorch.
- Data Loading pipeline using the ImageDataLoaders class provided by fastai
	- Here we make use of the get_image_files(path), which retrieves the paths to all image files within the specified path (path).
	- valid_pct=0.2: This parameter sets the percentage of the dataset to be used for validation (20% in this case).
	- label_func=RegexLabeller(pat = r'^([^/]+)_\d+'): This specifies a regular expression pattern (pat)
	  to extract labels from the filenames of the images. In this case, it extracts the class labels from the filenames based on the convention <class_name>_<index>.
	- item_tfms=Resize(224): This applies image transformations to each item (image) in the dataset.  In this case, it resizes each image to have a width and height of 224 pixels.
	```
	dls = ImageDataLoaders.from_name_func('.',
    get_image_files(path), valid_pct=0.2, seed=42,
    label_func=RegexLabeller(pat = r'^([^/]+)_\d+'),
    item_tfms=Resize(224))
	
	```
- Data display 

![alt text](https://github.com/vasanthgx/petdataset_classification/blob/main/images/fastai-1.png)

- Model Training with  **resnet34** function from fastai library
	- fine tuning with 3 epochs. We get the following error rates
	
	![alt text](https://github.com/vasanthgx/petdataset_classification/blob/main/images/er1.png)
	
- Next we train with a different model from the timm library - **'convnext_tiny_in22k'**
	- fine tuning with 3 epochs. We get better results than the previous one.
	
	![alt text](https://github.com/vasanthgx/petdataset_classification/blob/main/images/er2.png)
	
- Evaluation of the classifier 
	- with a basset hound breed of dog
	
	![alt text](https://github.com/vasanthgx/petdataset_classification/blob/main/images/basset.png)
	
	- prediction as below
	
	![alt text](https://github.com/vasanthgx/petdataset_classification/blob/main/images/99.png)
	
	
### Model Deployment

- Building a Gradio Application and hosting it on Huggingface/Spaces
	- Gradio is an open-source Python library that allows developers to quickly create and deploy interactive web-based interfaces for machine learning models. It simplifies the process of building user interfaces for ML applications, enabling users to interact with models through web browsers without requiring any knowledge of web development.
	- Within Hugging Face, Spaces refer to a platform for hosting interactive machine learning (ML) demo applications. It allows users to easily showcase their work, collaborate with others, and create a portfolio of their ML projects.
	```
	import gradio as gr
	image = gr.Image()
    label = gr.Label(num_top_classes=5)
	intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
	intf.launch()
	
	```
- Breed Classifier application

	![alt text](https://github.com/vasanthgx/petdataset_classification/blob/main/images/gradio3.png)
	



	




## Key Takeaways

 1) **Architecture Selection** : The choice of architecture plays a crucial role in the success of the project. U-Net architecture is well-suited for image segmentation tasks, especially when precise delineation of objects or regions is required. On the other hand, ResNet-34 is a popular choice for image classification tasks due to its balance between depth and computational efficiency.
2) **Data Preparation and Augmentation**: Proper data preparation and augmentation are essential for training robust models. In image segmentation, labeled data with accurate pixel-level annotations are required, while in breed classification, labeled images with breed annotations are necessary. Augmentation techniques such as rotation, scaling, and flipping can help increase the diversity of the training data and improve the generalization of the models.
3) **Transfer Learning and Pre-trained Models**: Leveraging pre-trained models can significantly accelerate the training process and improve model performance, especially when dealing with limited training data. Transfer learning techniques, such as fine-tuning pre-trained models like ResNet-34 using FastAI or Timm libraries, allow the models to adapt to the specific characteristics of the dataset while retaining the knowledge learned from large-scale datasets.
4) **Evaluation Metrics and Performance Analysis**: Choosing appropriate evaluation metrics is crucial for assessing the performance of the models accurately. For image segmentation tasks, metrics such as Intersection over Union (IoU) or Dice Coefficient are commonly used to measure the overlap between predicted and ground truth masks. For breed classification, metrics like accuracy, precision, recall, and F1-score are typically used to evaluate classification performance.
5) **Deployment and Integration**: Once the models are trained and evaluated, deploying them into production environments requires careful consideration of factors such as scalability, latency, and integration with existing systems. FastAI and Timm libraries provide deployment options such as exporting models to ONNX format or integrating them into web applications using frameworks like Flask or FastAPI. Additionally, TensorFlow Serving can be used for serving TensorFlow models in production environments.


## How to Run

The code is built on Google Colab on an iPython Notebook. 

```bash
Simply download the repository, upload the notebook and dataset on colab, and hit play!
```


## Roadmap

The next steps would be 

- Incorporate chosen features into model development.
- Train the model and assess its performance through rigorous evaluation.
- Fine-tune the model if necessary for optimization.
- Analyze model predictions for insights into the problem domain.
- Deploy the model and monitor its performance, iterating as needed for continuous improvement.


## Libraries 

**Language:** Python

**Packages:** Sklearn, Matplotlib, fastai, mobilenetv2, pix2pix, timm


## FAQ

### 1) What is Otsu Thresholding and  its significance in Object Segmentaion ?

[Thresholding is used to create a binary image from a grayscale image .](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_thresholding.html)
[More Details](https://en.wikipedia.org/wiki/Otsu's_method)

Otsu thresholding, named after Nobuyuki Otsu, is a popular method used for automatic image thresholding. The goal of thresholding is to segment an image into two parts: foreground and background, or object and background, based on pixel intensity. Otsu's method calculates an optimal threshold value by maximizing the between-class variance of the pixel intensities.

Here's a simplified explanation of how it works:

1) Initially, Otsu's method considers all possible threshold values between the minimum and maximum intensity levels in the image.
2) For each potential threshold value, it divides the pixels into two classes: those with intensities below the threshold and those with intensities above the threshold.
3) Then, it calculates the variances of these two classes.
4) Otsu's method selects the threshold value that maximizes the between-class variance, meaning it chooses the threshold where the difference in intensity between the two classes is the most significant.


The significance of Otsu thresholding in object segmentation lies in its ability to **automatically determine an optimal threshold value without requiring manual intervention**. This is particularly useful in scenarios where the image has varying lighting conditions or when the desired object has a distinct contrast with the background. By accurately separating the foreground object from the background based on intensity, **Otsu thresholding forms the foundation for many image processing tasks such as object detection, recognition, and tracking**. It simplifies the process and makes it more robust and adaptable to different types of images.







### 2) What are Neural Networks ?

Neural networks are a class of machine learning algorithms inspired by the structure and function of the human brain's biological neural networks. These artificial neural networks (ANNs) consist of interconnected nodes, called neurons, organized into layers. Each neuron receives input signals, performs a computation, and then passes the result to the neurons in the next layer.

Components of a neural network:

1) **Input Layer**: This layer receives the input data. Each neuron in this layer represents a feature of the input data.
2) **Hidden Layers**: These are intermediary layers between the input and output layers. Each neuron in a hidden layer takes input from the previous layer, performs a computation using weights and biases, applies an activation function, and passes the result to the neurons in the next layer. Deep neural networks have multiple hidden layers, hence the term "deep" learning.
3) **Output Layer**: This layer produces the final output of the network. The number of neurons in the output layer depends on the type of task the neural network is designed for. For example, in a binary classification task, there would typically be one neuron in the output layer, while in a multi-class classification task, there would be one neuron for each class.
4) **Weights and Biases**: Neural networks learn from data by adjusting the weights and biases associated with each neuron's connections. These parameters determine the strength of connections between neurons and affect the output of each neuron.
5) **Activation Functions**: Activation functions introduce non-linearities into the network, allowing it to learn complex patterns in the data. Common activation functions include ReLU (Rectified Linear Unit), Sigmoid, and Tanh.


**Neural networks are trained using optimization algorithms such as gradient** and its variants, which adjust the weights and biases to minimize a loss function. During training, the network learns to make predictions by iteratively updating its parameters based on the comparison between its predictions and the ground truth labels in the training data.

Neural networks, especially deep neural networks, have shown remarkable performance in various tasks such as image recognition, natural language processing, speech recognition, and many more, making them a cornerstone of modern artificial intelligence and machine learning.




### 3) Can you explain Convolutional Neural Networks ?

Convolutional Neural Networks (CNNs) are a specialized type of neural network designed specifically for processing structured grid data, such as images. They are highly effective for tasks like image recognition, object detection, and image classification. CNNs are inspired by the organization of the animal visual cortex, where individual neurons respond to specific stimuli in a restricted region of the visual field.

Components and concepts in CNNs:

1) **Convolutional Layers**: The fundamental building blocks of CNNs are convolutional layers. Each convolutional layer consists of a set of learnable filters (also known as kernels or convolutional kernels). These filters slide or convolve across the input image, performing element-wise multiplication with the pixel values in the region they cover and then summing up the results to produce a feature map. This process captures spatial hierarchies of patterns in the input image.
2) **Pooling Layers**: Pooling layers are used to downsample the feature maps generated by convolutional layers. The most common pooling operation is max-pooling, which selects the maximum value from a region of the feature map. Pooling helps in reducing the spatial dimensions of the feature maps, making the network more computationally efficient and reducing overfitting by extracting the most salient features.
3) **Activation Functions**: Activation functions, such as ReLU (Rectified Linear Unit), are applied after convolutional and pooling operations to introduce non-linearity into the network. This allows the CNN to learn complex patterns and relationships in the data.
4) **Fully Connected Layers**: Towards the end of the CNN architecture, one or more fully connected layers are typically used to perform classification or regression tasks. These layers connect every neuron in one layer to every neuron in the next layer, allowing the network to learn high-level representations of the input data.
5) **Convolutional Neural Network Architecture**: CNN architectures vary depending on the specific task and dataset. Common architectures include LeNet, AlexNet, VGGNet, GoogLeNet (Inception), ResNet, and more. These architectures differ in terms of the number of layers, types of layers, and the arrangement of those layers.
6) **Training**: CNNs are trained using backpropagation and optimization algorithms like stochastic gradient descent (SGD) or its variants. During training, the network learns to adjust the weights of its filters and fully connected layers to minimize a predefined loss function, typically categorical cross-entropy for classification tasks.




CNNs have revolutionized the field of computer vision and have achieved state-of-the-art performance on various image-related tasks, including image classification, object detection, image segmentation, and more. They have also been adapted for other types of structured data, such as time-series data and 1D signal processing.

### 4) What is Pix2Pix?

Pix2Pix is a type of conditional generative adversarial network (GAN) architecture that learns a mapping from an input image to an output image. Specifically, it is designed for image-to-image translation tasks, where the goal is to generate a corresponding output image based on a given input image. Pix2Pix was introduced by Phillip Isola et al. in their 2016 paper titled "Image-to-Image Translation with Conditional Adversarial Networks."

Overview of how Pix2Pix works:

1) **Conditional GAN Framework**: Pix2Pix extends the basic GAN framework by introducing a conditional setting. In a traditional GAN, the generator network takes random noise as input and generates fake images, while the discriminator network distinguishes between real and fake images. In Pix2Pix, both the generator and discriminator networks receive not only the generated/fake images but also the corresponding input images as conditional inputs.
2) **Generator Network**: The generator network in Pix2Pix takes an input image and produces an output image that is transformed in some way based on the input. This transformation could involve changing the style, color, texture, or any other characteristics of the input image. The generator typically consists of an encoder-decoder architecture, with convolutional layers for feature extraction and upsampling layers for increasing the resolution of the output image.
3) **Discriminator Network**: The discriminator network in Pix2Pix evaluates pairs of images (input image, corresponding output image) and distinguishes between real pairs (input-output pairs from the training dataset) and fake pairs (input image, generated output image pairs). The discriminator is trained to differentiate between real and fake pairs, while the generator is trained to fool the discriminator by generating realistic output images.
4) **Training Objective**: Pix2Pix uses an adversarial loss, computed by the discriminator, to encourage the generator to produce output images that are indistinguishable from real images. Additionally, Pix2Pix also employs a pixel-wise L1 loss between the generated output image and the ground truth output image to ensure that the generated images are visually similar to the target images.
5) **Applications**: Pix2Pix can be applied to various image-to-image translation tasks, such as converting sketches to color images, generating satellite images from map images, converting day-time images to night-time images, and more.

Overall, Pix2Pix has demonstrated impressive results in a wide range of image translation tasks, making it a powerful tool for image synthesis and manipulation.

### 5) What is U-Net arhitecture ?

U-Net is a popular convolutional neural network architecture designed for semantic segmentation tasks, particularly in medical image analysis, where precise pixel-level segmentation is required. It was introduced by Olaf Ronneberger, Philipp Fischer, and Thomas Brox in their [2015 paper titled](https://arxiv.org/abs/1505.04597) **"U-Net: Convolutional Networks for Biomedical Image Segmentation."**

The U-Net architecture is characterized by its symmetric encoder-decoder structure, which enables the network to capture both local and global features while preserving spatial information. Here's an overview of its key components:

1) **Encoder Path**: The encoder path is composed of multiple convolutional blocks followed by max-pooling layers. These convolutional blocks perform feature extraction, gradually reducing the spatial dimensions of the input image while increasing the number of feature channels. Each convolutional block typically consists of convolutional layers, activation functions (such as ReLU), and optionally batch normalization to stabilize training.
2) **Decoder Path**: The decoder path mirrors the encoder path but in reverse. It consists of up-sampling layers (often transposed convolutions or bilinear upsampling) followed by concatenation with feature maps from the corresponding encoder path. These concatenated feature maps are then passed through convolutional blocks, which gradually increase the spatial dimensions while decreasing the number of feature channels. The goal of the decoder path is to recover the spatial details lost during the encoding stage.
3) **Skip Connections**: One of the key innovations of U-Net is the extensive use of skip connections between the encoder and decoder paths. These skip connections directly connect feature maps from the encoder to the corresponding decoder layers at the same spatial resolution. By providing high-resolution feature maps from earlier layers to later layers, skip connections help the network preserve fine-grained details and improve segmentation accuracy.
4) **Final Layer**: The final layer of the U-Net architecture typically consists of a 1x1 convolutional layer followed by a softmax activation function. This layer generates the segmentation mask by predicting the probability of each pixel belonging to the target classes. The output mask has the same spatial dimensions as the input image, with each pixel assigned a class label or probability score.
5) **Loss Function**: U-Net is trained using a pixel-wise loss function, such as cross-entropy loss or Dice loss, which measures the discrepancy between the predicted segmentation mask and the ground truth mask. The network parameters are optimized using gradient descent-based algorithms, such as Adam or stochastic gradient descent (SGD).

U-Net has become a popular choice for various medical image segmentation tasks, including organ segmentation, tumor detection, cell segmentation, and more. Its symmetric architecture, skip connections, and pixel-wise loss function make it well-suited for tasks requiring precise and detailed segmentation of structures within images.


## Acknowledgements


 - [Tensorflow datasets](https://www.tensorflow.org/datasets/overview)
 - [Image Segmentaion](https://www.tensorflow.org/tutorials/images/segmentation)
 - [Gradio crash course](https://www.youtube.com/watch?v=eE7CamOE-PA&list=LL&index=2)
 - [TensorFlow tutorials](https://www.youtube.com/playlist?list=PLQY2H8rRoyvwWuPiWnuTDBHe7I0fMSsfO)


## Contact

If you have any feedback/are interested in collaborating, please reach out to me at vasanth_1627@gmail.com


## License

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

