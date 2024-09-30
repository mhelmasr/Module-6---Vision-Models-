# Introduction to Image Classification using Convolutional Neural Networks (CNNs)

## Lesson Overview :pencil2:

This lesson introduces Convolutional Neural Networks (CNNs), a powerful deep learning architecture optimized for image processing and classification tasks. Participants will explore CNN architecture and functionality, understanding why these networks excel over traditional neural networks in handling visual data. The lesson begins with an overview of CNN architecture, explaining how these networks mimic the human visual cortex to process visual information. Through hands-on exercises, Participants will implement and train a CNN using popular deep learning frameworks, gaining practical experience in model design, hyperparameter tuning, and performance evaluation for image classification tasks. This comprehensive introduction will equip students with the knowledge to leverage CNNs effectively in various computer vision applications.

<br>  <!-- don't remove -->

## Learning Objectives :notebook:

By the end of this lesson, you will be able to:

1. Learning Objective 1: Define the key components of Convolutional Neural Networks (CNNs) and their role in image classification.
2. Learning Objective 2: Explain the advantages of CNNs over regular neural networks in processing visual data.
3. Learning Objective 3: Implement essential image preprocessing techniques and transformations required for CNN input.
4. Learning Objective 4: Differentiate between various types of convolutional layers and their specific functionalities within a CNN architecture.
5. Learning Objective 5: Design and tune a CNN model to solve an image classification problem and achieve the highest possible performance.

<br>

## Key Definitions and Examples :key:

### Convolutional Neural Network

A Convolutional Neural Network (CNN), also known as ConvNet, is a specialized type of deep learning algorithm mainly designed for tasks like object recognition, including image classification, detection, and segmentation.

![image](https://github.com/user-attachments/assets/cf93a607-f366-4502-8b4f-482d0a037c4d)

<br>  <!-- don't remove -->

#### Applications of Convolutional Neural Network:

1. *Image Classification* – Search Engines, Social Media, Recommender Systems
2. *Face Recognition Applications* - Social Media, Identification, and Surveillance
3. *Medical Image Computing* – Predictive Analytics, Healthcare Data Science
   
<br>  <!-- don't remove -->

### Key Differences between Neural Network and Convolutional Neural Networks :

- **Architecture**: ANNs use fully connected layers where each neuron connects to every neuron in the previous layer, whereas CNNs utilize convolutional layers that preserve spatial structure by connecting neurons locally.
- **Feature Extraction**: ANNs treat each pixel as a separate feature, requiring more manual intervention for complex tasks. CNNs automatically extract relevant spatial features using filters.
- **Efficiency and Scalability**: CNNs, through parameter sharing and localized operations, are significantly more scalable and efficient, especially for large and complex images, while ANNs require more computational resources and struggle with scalability.
- **Accuracy and Transfer Learning**: CNNs tend to outperform ANNs in accuracy, especially on complex tasks, and are better suited for transfer learning due to their ability to capture hierarchical spatial features.


<br>  <!-- don't remove -->

### Convolutional Neural Network Architecture

CNNs function as expert pattern detectors. Unlike traditional neural networks, where every input value directly connects to a neuron, CNNs employ a series of operations inspired by the way our visual cortex works.
The Classical CNN Architecture:
- Convolutional layers apply filters to input images, extracting features through convolution operations. 
- Pooling layers down-sample feature maps, reducing computational complexity. 
- Fully connected layers integrate features for classification.

![image](https://github.com/user-attachments/assets/575f0c81-f062-4f72-ae09-5e3d29ffa0f7)

<br>  <!-- don't remove -->

#### A Classical Convolutional Neural Network Architecture

```python

import tensorflow as tf
from keras import layers, models, datasets

# Simple CNN Model Architecure Design
model = models.Sequential([
                          layers.Conv2D(32, kernel_size=(4, 4), activation='relu', input_shape=(28, 28,1)),
                          layers.MaxPool2D(2,2),
                          layers.Flatten(),
                          layers.Dense(16, activation="relu"),
                          layers.Dense(8, activation="relu"),
                          layers.Dense(10, activation="softmax")
                          ])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

### Convolutional Neural Network Layers

 1. **Convolutional Layers:**
    - These are the core building blocks of a CNN. The convolutional layer applies a number of filters to the input. Each filter detects different features such as edges, colors, or more complex shapes. The output of this layer is called a feature map, which highlights the areas of the input image most activated by the filter.
 
 2. **Pooling Layers:**
    - Pooling (also known as subsampling or downsampling) reduces the dimensionality of each feature map but retains the most important information. Max pooling is a common technique used to reduce the spatial dimensions of the input volume for the next convolution layer. It works by selecting the maximum value from each cluster of neurons at the prior layer.
 
 3. **Fully Connected Layers:**
    - After several convolutional and pooling layers, the high-level reasoning in the neural network is done via fully connected layers. Neurons in a fully connected layer have full connections to all activations in the previous layer. This part of the network is typically responsible for assembling the features extracted by the convolutional layers and pooling layers to form the final outputs.
 
 4. **Output Layer:**
    - The final layer uses an activation function such as softmax (for classification tasks) to map the output of the last fully connected layer to probability distributions over classes.


<br>  <!-- don't remove -->


## Additional Resources :clipboard: 

If you would like to study these concepts before the class or would benefit from some remedial studying, please utilize the resources below:

- [Interactive CNN Explainer](https://deeplizard.com/resource/pavq7noze2)
- [Convolutional Neural Networks: A Brief History of their Evolution](https://medium.com/appyhigh-technology-blog/convolutional-neural-networks-a-brief-history-of-their-evolution-ee3405568597)
