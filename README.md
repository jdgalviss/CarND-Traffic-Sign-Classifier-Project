# **Traffic Sign Recognition** 

## Writeup

### In this project,we present a pipeline to classify the signs defined in the German Traffic Sign dataset using convolutional Neural Networks.

---

**Project goals**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Preprocess data
    * Augment data
    * Normalize data
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_imgs/01_histogram_train_data.png "Histogram"
[image2]: ./output_imgs/01_histogram_valid_data.png "Histogram Validation"
[image3]: ./output_imgs/01_histogram_test_data.png "Histogram Test"
[image4]: ./output_imgs/02_all_classes.png "All classes"
[image5]: ./output_imgs/03_augmented.png "Data Augmentation"
[image6]: ./test_images/test/120.jpeg "test image 1"
[image7]: ./test_images/test/18.jpeg "test image 2"
[image8]: ./test_images/test/1.jpeg "test image 3"
[image9]: ./test_images/test/3.jpeg "test image 4"
[image10]: ./test_images/test/7.jpeg "test image 5"
[image11]: ./output_imgs/07_test1.png "softmax test 1"
[image12]: ./output_imgs/07_test8.png "softmax test 2"
[image13]: ./output_imgs/07_test7.png "softmax test 3"
[image14]: ./output_imgs/07_test9.png "softmax test 4"
[image15]: ./output_imgs/07_test10.png "softmax test 5"
[image16]: ./output_imgs/05_train_accuracy.png "train accuracy"
[image17]: ./output_imgs/05_train_loss.png "train loss"
[image18]: ./output_imgs/05_validation_accuracy.png "validation accuracy"
[image19]: ./output_imgs/05_architecture.png "architecture"
[image20]: ./output_imgs/accuracy.png "accuracy"
[image21]: ./output_imgs/06_visualization1.png "visualization1"
[image22]: ./output_imgs/06_visualization2.png "visualization2"
[image23]: ./output_imgs/06_visualization3.png "visualization3"
[image24]: ./output_imgs/06_visualization4.png "visualization4"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README


You're reading it! and here is a link to my [project code](https://github.com/jdgalviss/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Dataset Summary.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 pixels
* The number of unique classes/labels in the data set is 43

#### 2. Dataset Exploration
The following figure shows one image per each one of the 43 classes included in the dataset.


![alt text][image4]

The following bar charts show how the data is distributed:


![alt text][image1]

![alt text][image2]

![alt text][image3]

We can see how the data is unbalanced, i.e. we don't have the same ammount of images for each class, this characteristic of the dataset (mainly the training data set) can be problematic, since the network will be biased towards the classes with the most images. We'll take this into account when we build our model architecture.


### Design and Test a Model Architecture

#### 1. Data Preprocessing

As a first step, the data is augmented offline in order to have a bigger dataset (3x), by apllying the following operations:
* Rotate a random angle between 15 and -15 degrees
* Translate randombly betwwen -5 and 5 pixels both vertically and horizontally
* Scale Image
* Change Brightness and Saturation

Following image, shows the augmentation procedure run on every image:

![alt text][image5]

By having more data, considering the rotation, translation, scaling and the changes in brightness and saturation, we expect to have a more varied source of information, such that our model can generalize to all the ways data can be presented in.

Even though different color representations like YUV were tested, results with RGB were the same, this is why no color transformation was implemented

#### 2. Model architecture

The model (same as LeNet) consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Dropout				|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 			     	|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU					|												|
| Dropout				|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 	      			|
| Flatten   	      	| make sure output is 1-D 	      			    |
| Fully connected		| 120 outputs        							|
| RELU					|												|
| Dropout				|												|
| Fully connected		| 84 outputs        							|
| RELU					|												|
| Dropout				|												|
| Fully connected		| 43 outputs - logits        					|
| Softmax				|            									|

![alt text][image19]

#### 3. Model Training

**Loss Function**
The same dropout, L2 Regularization is implemented in order to avoid over fitting, this term is added to the loss function:
```python
loss_operation = tf.reduce_mean(cross_entropy) + regularization_factor*tf.nn.l2_loss(conv1_W) + regularization_factor*tf.nn.l2_loss(conv2_W) + regularization_factor*tf.nn.l2_loss(fc1_W) + regularization_factor*tf.nn.l2_loss(fc2_W) + regularization_factor*tf.nn.l2_loss(fc3_W) + regularization_factor*tf.nn.l2_loss(conv1_b) + regularization_factor*tf.nn.l2_loss(conv2_b) + regularization_factor*tf.nn.l2_loss(fc1_b) + regularization_factor*tf.nn.l2_loss(fc2_b) + regularization_factor*tf.nn.l2_loss(fc3_b)
```
On the other hand, as discussed before, data is unbalanced, for this reason we are going to weight logits during training, such that classes with the less number of images have a bigger weight and vice-versa:

```python
for i in range(0,43,1):
    weights_arr.append(math.log(X_train_augmented.shape[0]/max(counted_train[i],1))+1)
...
weighted_logits = tf.multiply(logits,weights_loss)
...

```
The optimizer used is RMSProp, an algorithm which adjusts the learning rate in time according to a decay parameter. It uses a moving average of squared gradients to normalize the gradient itself.

```python
optimizer = tf.train.RMSPropOptimizer(decay=0.95, epsilon=0.1, learning_rate=rate)
```

#### 4. Approach
Like already stated, several techniques were implemented to get a higher accuracy and avoid overfitting with LeNet architecture:
* Data augmentation
* Dropout
* L2 Regularization
* Weighted logits for unbalanced data

My final model results were:
* training set accuracy of 99%
* validation set accuracy of 97% 
* test set accuracy of 95%

Tensorboard was used to visualize the training process, here we can see how the accuracies and loss changed during training.

![alt text][image16]
![alt text][image17]
![alt text][image18]

![alt text][image20]

* What was the first architecture that was tried and why was it chosen?
For simplicity, LeNet architecture was chosen, even though first tests didn't show a validation accuracy of more than 0.9, after implemented the previous mentioned techniques, the accuracy was strongly improved.
* What were some problems with the initial architecture?
Without droput or any type of regularization, it is quite likely that the gradient could die or that the model would tend to overfit. On the other hand, without data augmentation, when testing the model on external data, it wouldnt get an accuracy of more than 0.8, which shows that the model wasn't generalizing well, for this, regularization and data augmentation were key.
* How was the architecture adjusted and why was it adjusted? The only change made to the architecture was adding droput to each layer of the network to avoid overfitting.
* Which parameters were tuned? How were they adjusted and why?
* Learning rate and decay: these two play a really important role in th RMSProp algorithm, at first two high values in the learning rate would make the model's accuracy get stucked.
* Batches number: it was set to 128 after several tests
* L2 Regularization: if the weight of l2 regularization is too high, it doesn't let the model adjust to training data, a small value wasof 0.001 was chosen.
* Epochs: After watching the accuracy and loss plots, it was clear that 10 epochs were not enough and the model could keep on learning, this value was finally set to 70 epochs, even though "early stopping" should also be considered to avoid overfitting.


### Testing the Model on New Images

#### 1. Choosing 5 german traffic signs images.

Here, five traffic signs images were chosen:

![alt text][image6] ![alt text][image7] ![alt text][image8] 
![alt text][image9] ![alt text][image10]

These images might present following difficulties to the model:
* First image: It presents a circular object behind the sign, whose shape differs from the actual sign.
* Second image: the little sign under the traffic sign might confuse the classifier.
* Third image: It wasn't taken in front of the sign, but from a different perspective, so the number and the sign shapes are distorted.
* Fourth image: The sign is rectangular, insted of circular as usual
* Fifth image: It was also taken from a different perspective


#### 2. Model's predictions

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Priority road    		| Priority road  								| 
| No entry     			| No entry 										|
| Speed limit (30km/h)	| Speed limit (30km/h)							|
| Speed limit (60km/h)	| Speed limit (60km/h)         				|
| Speed limit (100km/h)	| Speed limit (100km/h)							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. A really good result, since despite being a small sample, it confirms the accuracies obtained for training, validation and test data.

#### 3. How certain was the model?

The code for making predictions on my final model is located in the 102th cell of the Ipython notebook.

The following images show how the softmax probabilities were distributed for the test images:

![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]

The model was extremly certain on test images.

### Visualizing the Neural Network
Even though Neural Networks are sometimes seen as black boxes, by showing feature maps we can see how they treat information to extract features which become more complex as the network depth grows. In order to analyze this, it is possible to get following visualizations of the feature maps for the first (first 6 feature maps) and second (next 16 feature maps) convolution layer.

![alt text][image21]
* conv1: In this case, the feature map 4 highlights the yellow in the middle of the traffic signs, it also seen how the rest highlight the edges of the sign which are located at 45 degrees.
* conv2: feature maps seem to focus on more and more on determined edges

![alt text][image22]
* conv1: In this case, the feature map 4 highlights the red in the stop sign, the feature map 5 highlights the edges of the letters and the sign itsself.

![alt text][image23]
* conv1: In this case, the feature map 0 highlights the blue in the traffic sign while the rest seem to be focused on the edges in different directions.
  
![alt text][image24]
* conv1: In this case, the feature map 2 highlights the edges corresponding to the number. However the feature maps 

The conv2 feature maps represent a higher extraction of the images, hence it is a little more difficult to distinguish their meaning
