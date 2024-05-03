

![logo](https://github.com/vasanthgx/petdataset_classification/blob/main/images/logo.gif)


# Project Title


**Image Segmentaion with Breed Classification ML Project**
 <img src="https://github.com/Anmol-Baranwal/Cool-GIFs-For-GitHub/assets/74038190/b3fef2db-e671-4610-bb84-1d65533dc5fb" width="300" align='right'>

<br><br>


# Table of Contents
1) [Introduction](##Introduction)
2) [Project Overview](##ProjectOverview)
3) [Key Features](##Key Features)
4) [FAQ](FAQ)



## Introduction

This project aims to demonstrate the application of machine learning techniques in two key areas of computer vision: image segmentation and breed classification. Image segmentation involves partitioning an image into multiple segments or regions to simplify its representation and facilitate further analysis. Breed classification, on the other hand, focuses on identifying the breed of animals depicted in images, in our case, dogs.


## Project Overview

n this project, we have developed a machine learning pipeline that combines state-of-the-art techniques for image segmentation and breed classification. The pipeline processes input images of dogs, performs semantic segmentation to identify distinct regions within the images, and subsequently classifies the breed of the dog present in each segmented region.

## Key Features

- **Image Segmentation**: We utilize advanced convolutional neural network (CNN) architectures to perform semantic segmentation, enabling precise delineation of different objects or regions within the input images.
- **Breed Classification**: Leveraging transfer learning, we fine-tune pre-trained CNN models to classify the breed of dogs present in the segmented regions. This allows us to achieve high accuracy even with limited training data.
- **End-to-End Pipeline**: Our project provides a seamless end-to-end solution for image segmentation and breed classification, enabling users to input raw images and obtain detailed segmentation masks along with breed predictions.
- **Model Deployment**: [We have deployed the project in HuggingSpaces through Gradio Application.](https://huggingface.co/spaces/Vasanthgx/oxford_pets_breed_classification)

![alt text](https://github.com/vasanthgx/petdataset_classification/blob/main/images/gradio.png)

## Implementation Details

- Dataset: The Oxford-IIIT Pet Dataset (view below for more details)
- Model: [MobileNetV2]('https://keras.io/api/applications/mobilenet/')
- Input: 37 category pet dataset with roughly 200 images for each class
- Output: Segmentation mask , Breed classification

## Dataset Details

[This dataset was obtained from this repository](https://www.robots.ox.ac.uk/~vgg/data/pets/)

The Oxford-IIIT Pet Dataset is a widely used collection of images containing various breeds of cats and dogs. It was created by the Visual Geometry Group at the University of Oxford and the IIIT Delhi. The dataset consists of over 7,000 images of pets belonging to 37 different categories, with each category representing a specific breed.
Each image in the dataset is annotated with bounding boxes and class labels, making it suitable for tasks such as object detection and classification. The dataset provides a diverse range of poses, backgrounds, and lighting conditions, making it valuable for training and evaluating computer vision algorithms.
Researchers and practitioners often use the Oxford-IIIT Pet Dataset for tasks such as fine-grained classification, instance segmentation, and pose estimation. Its availability and richness make it a benchmark dataset in the field of computer vision, contributing to advancements in pet recognition technology and beyond.

### Dog and Cat Breeds in the dataset

 ![alt text](https://github.com/vasanthgx/petdataset_classification/blob/main/images/dataset_stats.png)

### Annotation Examples from the dataset

 ![alt text](https://github.com/vasanthgx/petdataset_classification/blob/main/images/annotation_examples.png)


## Evaluation and Results

### Downloading and unzipping the dataset
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

#### What is Image Segmentaion?( more details in FAQ section)
In an image classification task, the network assigns a label (or class) to each input image. However, suppose you want to know the shape of that object, which pixel belongs to which object, etc. In this case, you need to assign a class to each pixel of the image—this task is known as segmentation. A segmentation model returns much more detailed information about the image. Image segmentation has many applications in medical imaging, self-driving cars and satellite imaging, just to name a few.



- [Loading Dataset and Exploration](https://www.tensorflow.org/datasets/catalog/oxford_iiit_pet)


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
  image = tf.cas
```

- Model Building




### Correlation between the features
  Correlation tests are often used in feature selection, where the goal is to identify the most relevant features (variables) for a predictive model. Features with high correlation with the target variable are often considered important for prediction. However, it's essential to note that correlation does not imply causation, and other factors such as domain knowledge and data quality should also be considered in feature selection.

![alt text](https://github.com/vasanthgx/traffic_prediction/blob/main/images/correlation_graph_initial_dataset.png)

**As we can see, there is no strong correlation between the features**

## Data Cleaning and Pre Processing
- **Pre-processing (Cleaning): Address missing (NULL) values - drop or imputation.**
    - **we will use the ffill() method**
    ```
    data.ffill(inplace = True)
    ```
	
- **Since we have already seen poor reperesentation of 'snow_1h' and 'rain_1h', and similarity between weather_main and  'weather_description' we will drop the three features for the model.**
    ```
    data1 = data.drop(['snow_1h', 'rain_1h','weather_description'] , axis =1)
    ```
- **Converting 'holiday' feature into just holiday and 'unknown'.**
    ```
    data1['holiday'] = data1['holiday'].apply(lambda x: 'unknown' if pd.isna(x) else 'holiday' ) 
    ```
- **Next we will first convert the 'date_time' feature into a pandas datetime object.**
    ```
    data1['date_time'] = pd.to_datetime(data1['date_time'], format = '%d-%m-%Y %H:%M')
    ```
- **We now extract the 'year', 'month', 'weekday' and 'hour' from the datetime object.**
    ```
    data1['year'] = data1['date_time'].dt.year
    data1['month'] = data1['date_time'].dt.month
    data1['weekday'] = data1['date_time'].dt.weekday
    data1['hour'] = data1['date_time'].dt.hour
    ```
- **Next we will now divide the 24 hours of the day into 'before_sunrise', 'after_sunrise', 'afternoon' and 'night' categories.**
    ```
    data1['hour'].unique()
    ```
- **We will create a function ,which will split the hours into the above four categories.**
    ```
    def day_category(hour):
        day_section = ''
        if hour in [1,2,3,4,5]:
            day_section = 'before_sunrise'
        elif hour in [6,7,8,9,10,11,12]:
            day_section = 'after_sunrise'
        elif hour in [13,14, 15, 16, 17, 18]:
            day_section = 'evening'
        else :
            day_section = 'night'
        return day_section
    ```
- **Using the map() function to loop through the 'hour' feature and based on the hour - value we will allot the 4 day-sections. This way we will create one more feature 'day_section' in our existing dataset.**
    ```
    data1['day_section'] = data1['hour'].map(day_category)
    ```
- **Next we use the pd.get_dummies function to do one hot encoding of the categorical features 'holiday', 'weather_main' and 'day section'.**
    ```
    data1 = pd.get_dummies(data1, columns =['holiday', 'weather_main','day_section'])
    ```
- **Finally we set the feature 'date_time' as row index in our dataset.**
    ```
    data1.set_index('date_time',inplace = True)
    ```
### Correlation testing - second time
- **After the above feature engineering.**

    ```
    corr_data1 = data1.corr()
    fig, ax = plt.subplots(figsize = (15, 10))
    plt.xticks(rotation =45)
    sns.heatmap(corr_data1, annot = True, linewidths = .5, fmt = '.1f', ax = ax)
    plt.show()
    ```
![alt text](https://github.com/vasanthgx/traffic_prediction/blob/main/images/correlation_graph_after_feature_engineering.png)

![alt text](https://github.com/vasanthgx/traffic_prediction/blob/main/images/caption.png)

## Feature Importance and Selection Using Random Forest Regressor

Feature importance and selection with the Random Forest Regressor involve identifying the most influential features in predicting the target variable.

**Feature Importance:** Random Forest Regressor calculates feature importance based on how much the tree nodes that use that feature reduce impurity across all trees in the forest. Features that lead to large reductions in impurity when used at the root of a decision tree are considered more important. Random Forest assigns a score to each feature, indicating its importance. Higher scores signify more important features.

**Visualizing Feature Importance:** Plotting the feature importance scores can provide insights into which features are most relevant for prediction. This visualization can aid in understanding the data and making decisions about feature selection.

![alt text](https://github.com/vasanthgx/traffic_prediction/blob/main/images/feature_selection.png)

In summary, feature importance and selection with Random Forest Regressor involve identifying and prioritizing features based on their contribution to predicting the target variable. This process can enhance model performance, interpretability, and understanding of the underlying data.

## Model Development

- **we will select just the top 7 features that we got from the Random Forest Regressor**
    ```
    important_features = [ 'hour','temp','weekday','day_section_night','month', 'year','clouds_all']
    ```
- Splitting the dataset into training and **validation set**. This validation set is to test our model internally before submitting it to the test set
    - *Note : we have already been provided the test data set for the hackathon*

- **Scaling : we do the scaling of the data using the StandardScaler() function from sklearn**

- **Experimenting with different models , so that we can select the best model for our submision**

![alt text](https://github.com/vasanthgx/traffic_prediction/blob/main/images/experimenting_models.png)

- **Selecting the best model**
    ```
    regrh = HistGradientBoostingRegressor(random_state=32)
    regrh.fit(x_train_scaled, y_train)
    y_pred = regrh.predict(x_test_scaled)
    print(f"r2 score : {r2_score(y_test, y_pred)} \n mean squared error : {mean_squared_error(y_test, y_pred)} \n mean absolute error : {mean_absolute_error(y_test,y_pred)} ")
    ```

## Testing and Creating Output CSV

- **we repeat the same process of data cleaing, pre processing, scaling etc with the test data.**
- **finally we submit the submission file.**


## Key Takeaways

After the hackathon process, key takeaways include:

1. **Data Exploration is Crucial**: Understanding the dataset thoroughly is essential before building any machine learning model. Exploratory data analysis helps in identifying patterns, outliers, and relationships within the data.

2. **Feature Engineering Matters**: Creating meaningful features from the existing data can significantly improve model performance. Techniques like encoding categorical variables, creating new features from datetime data, and scaling numerical features might be beneficial.

3. **Model Selection and Tuning**: Experimenting with various machine learning algorithms and hyperparameters can lead to improved performance. Techniques like cross-validation and hyperparameter tuning help in selecting the best model configuration.

4. **Evaluation Metrics**: Choosing the right evaluation metric based on the problem domain is crucial. In regression tasks like traffic volume prediction, metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), or Root Mean Squared Error (RMSE) are commonly used to evaluate model performance.

5. **Interpreting Model Results**: Understanding how the model makes predictions and the importance of different features can provide valuable insights into the problem domain. Techniques like feature importance analysis help in understanding which features contribute most to the model's predictions.

6. **Continuous Learning**: Hackathons are great learning experiences, and reflecting on what worked well and what could be improved prepares you for future challenges. Continuous learning and experimentation are key to mastering machine learning techniques.

Overall, participating in hackathons provides valuable hands-on experience in solving real-world problems using machine learning techniques. 


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

**Packages:** Sklearn, Matplotlib, Pandas, Seaborn


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

### 4) What is U-Net arhitecture ?

U-Net is a popular convolutional neural network architecture designed for semantic segmentation tasks, particularly in medical image analysis, where precise pixel-level segmentation is required. It was introduced by Olaf Ronneberger, Philipp Fischer, and Thomas Brox in their [2015 paper titled](https://arxiv.org/abs/1505.04597) **"U-Net: Convolutional Networks for Biomedical Image Segmentation."**

The U-Net architecture is characterized by its symmetric encoder-decoder structure, which enables the network to capture both local and global features while preserving spatial information. Here's an overview of its key components:

1) **Encoder Path**: The encoder path is composed of multiple convolutional blocks followed by max-pooling layers. These convolutional blocks perform feature extraction, gradually reducing the spatial dimensions of the input image while increasing the number of feature channels. Each convolutional block typically consists of convolutional layers, activation functions (such as ReLU), and optionally batch normalization to stabilize training.
2) **Decoder Path**: The decoder path mirrors the encoder path but in reverse. It consists of up-sampling layers (often transposed convolutions or bilinear upsampling) followed by concatenation with feature maps from the corresponding encoder path. These concatenated feature maps are then passed through convolutional blocks, which gradually increase the spatial dimensions while decreasing the number of feature channels. The goal of the decoder path is to recover the spatial details lost during the encoding stage.
3) **Skip Connections**: One of the key innovations of U-Net is the extensive use of skip connections between the encoder and decoder paths. These skip connections directly connect feature maps from the encoder to the corresponding decoder layers at the same spatial resolution. By providing high-resolution feature maps from earlier layers to later layers, skip connections help the network preserve fine-grained details and improve segmentation accuracy.
4) **Final Layer**: The final layer of the U-Net architecture typically consists of a 1x1 convolutional layer followed by a softmax activation function. This layer generates the segmentation mask by predicting the probability of each pixel belonging to the target classes. The output mask has the same spatial dimensions as the input image, with each pixel assigned a class label or probability score.
5) **Loss Function**: U-Net is trained using a pixel-wise loss function, such as cross-entropy loss or Dice loss, which measures the discrepancy between the predicted segmentation mask and the ground truth mask. The network parameters are optimized using gradient descent-based algorithms, such as Adam or stochastic gradient descent (SGD).

U-Net has become a popular choice for various medical image segmentation tasks, including organ segmentation, tumor detection, cell segmentation, and more. Its symmetric architecture, skip connections, and pixel-wise loss function make it well-suited for tasks requiring precise and detailed segmentation of structures within images.


## Acknowledgements


 - [Metro Interstate Traffic](https://www.kaggle.com/datasets/rgupta12/metro-interstate-traffic-volume/code)
 - [Finding indicators for high traffic volume](https://www.kaggle.com/code/roberttareen/finding-indicators-for-high-traffic-on-i-94)
 - [Time Series Analysis](https://www.analyticsvidhya.com/blog/2021/10/a-comprehensive-guide-to-time-series-analysis/)


## Contact

If you have any feedback/are interested in collaborating, please reach out to me at vasanth_1627@gmail.com


## License

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

