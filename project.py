#========================= Load pickled data ==================
import pickle
import numpy as np
import tensorflow as tf

#  Fill this in based on where you saved the training and testing data

training_file = 'traffic-signs-data/train.p'
training_file_augmented = 'traffic-signs-data/train_augmented.p'
validation_file='traffic-signs-data/valid.p'
testing_file = 'traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(training_file_augmented, mode='rb') as f:
    train_augmented = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_train_augmented, y_train_augmented = train_augmented['features'], train_augmented['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
import matplotlib.pyplot as plt
# plt.imshow(X_train_augmented[10])
# plt.show()
# plt.imshow(X_train[10])
# plt.show()
# plt.imshow(X_valid[10])
# plt.show()
#===========================Data info ===================
import pandas
### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

#  Number of training examples
n_train = X_train.shape[0]

#  Number of training augmented examples
n_train_augmented = X_train_augmented.shape[0]

#  Number of validation examples
n_validation = X_valid.shape[0]

#  Number of testing examples.
n_test = X_test.shape[0]

#  What's the shape of an traffic sign image?
image_shape = X_train.shape[1:3]

#  How many unique classes/labels there are in the dataset.
n_classes = max(y_train)

print("Data info...")
print("Number of training examples =", n_train)
print("Number of training examples =", n_train_augmented)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)



#================Transform images to YUV color space============
print("Transforming images to YUV color space")
import cv2
#plt.imshow(X_valid[indices[22]])
#plt.show()
X_train_yuv = np.copy(X_train)
X_valid_yuv = np.copy(X_valid)
X_test_yuv = np.copy(X_test)

if(False):
    i = 0
    for image in X_train:
        X_train[i] = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        i = i +1

    i = 0
    for image in X_train_augmented:
        X_train_augmented[i] = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        i = i +1

    i = 0
    for image in X_valid:
        X_valid[i] = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        i = i +1

    i = 0
    for image in X_test:
        X_test[i] = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        i = i +1


# ========================== Normalize data ===========================
print("Normalizing data...")
X_train, y_train = train['features'], train['labels']  

X_train_norm = (X_train - X_train.mean()) / (np.max(X_train) - np.min(X_train))
X_valid_norm = (X_valid - X_train.mean()) / (np.max(X_train) - np.min(X_train))
X_test_norm = (X_test - X_train.mean()) / (np.max(X_train) - np.min(X_train))
X_train_augmented_norm = (X_train_augmented - X_train_augmented.mean()) / (np.max(X_train_augmented) - np.min(X_train_augmented))
#X_valid_norm = (X_valid - X_train_augmented.mean()) / (np.max(X_train_augmented) - np.min(X_train_augmented))
#X_test_norm = (X_test - X_train_augmented.mean()) / (np.max(X_train_augmented) - np.min(X_train_augmented))

#X_train_augmented_norm =  2.0*(X_train_augmented - X_train_augmented.min())/(X_train_augmented.max()-X_train_augmented.min()) -1.0 
#X_valid_norm = 2.0*(X_valid - X_valid.min())/(X_valid.max()-X_valid.min()) -1.0
#X_test_norm = 2.0*(X_test - X_test.min())/(X_test.max()-X_test.min()) -1.0
#X_train_norm = 2.0*(X_train - X_train.min())/(X_train.max()-X_train.min()) -1.0

#X_train_augmented_norm = (X_train_augmented - X_train_augmented.min())/(X_train_augmented.max()-X_train_augmented.min())
#X_valid_norm = (X_valid - X_valid.min())/(X_valid.max()-X_valid.min())
#X_test_norm = (X_test - X_test.min())/(X_test.max()-X_test.min())
#X_train_norm = (X_train - X_train.min())/(X_train.max()-X_train.min())

#shuffle data
from sklearn.utils import shuffle
print("Shuffling data...")
X_train_augmented_norm, y_train_augmented = shuffle(X_train_augmented_norm, y_train_augmented)
X_train_norm, y_train= shuffle(X_train_norm, y_train)

#===============Define Model Architecture========
from tensorflow.contrib.layers import flatten
tf.reset_default_graph()
mu = 0
sigma = 0.1
keep_prob = tf.placeholder(tf.float32, name = 'droput_keep_probability')
conv1_W = tf.Variable(tf.truncated_normal(name = 'weights_conv1', shape=(5, 5, 3, 6), mean = mu, stddev = sigma))
conv1_b = tf.Variable(tf.zeros(6))
conv2_W = tf.Variable(tf.truncated_normal(name = 'weights_conv2', shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
conv2_b = tf.Variable(tf.zeros(16))
fc1_W = tf.Variable(tf.truncated_normal(name = 'weights_fc1', shape=(5*5*16, 120), mean = mu, stddev = sigma))
fc1_b = tf.Variable(tf.zeros(120))
fc2_W  = tf.Variable(tf.truncated_normal(name = 'weights_fc2', shape=(120, 84), mean = mu, stddev = sigma))
fc2_b  = tf.Variable(tf.zeros(84))
fc3_W  = tf.Variable(tf.truncated_normal(name = 'weights_fc3', shape=(84, 43), mean = mu, stddev = sigma))
fc3_b  = tf.Variable(tf.zeros(43))
def LeNet(x):
    # conv 1   
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID', name = 'conv1')
    conv1 = tf.nn.bias_add(conv1, conv1_b, name = 'bias_conv1')
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.dropout(conv1, keep_prob, name = 'dropout1')
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    # conv2
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID', name = 'conv2')
    conv2 = tf.nn.bias_add(conv2, conv2_b, name = 'bias_conv2')
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.dropout(conv2, keep_prob, name = 'dropout2')
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    # flatten
    fc0   = flatten(conv2)
    # fully connected 1
    fc1  = tf.matmul(fc0, fc1_W) + fc1_b
    fc1    = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob, name = 'dropout3')

    # fully connected 2
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    fc2    = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, keep_prob, name = 'dropout4')

    # output
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    return logits

#Features and labels
x = tf.placeholder(tf.float32, (None, 32, 32, 3), name = 'Input')
y = tf.placeholder(tf.int32, (None), name = 'Output')
one_hot_y = tf.one_hot(y, 43)

#weights for loss
import math
from collections import Counter
counted_train = Counter(y_train_augmented)
print(counted_train)
weights_arr = []
for i in range(0,43,1):
    #weights_arr.append( 1.0 - (counted_train[int(i)]/4020)*0.8)
    weights_arr.append(math.log(X_train_augmented.shape[0]/max(counted_train[i],1))+1)
    

weights_loss = tf.constant(weights_arr)

rate = 0.02
regularization_factor = 0.0001
logits = LeNet(x)
weighted_logits = tf.multiply(logits,weights_loss)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy) + regularization_factor*tf.nn.l2_loss(conv1_W) +regularization_factor*tf.nn.l2_loss(conv2_W) +regularization_factor*tf.nn.l2_loss(fc1_W) +regularization_factor*tf.nn.l2_loss(fc2_W) +regularization_factor*tf.nn.l2_loss(fc3_W) +regularization_factor*tf.nn.l2_loss(conv1_b) +regularization_factor*tf.nn.l2_loss(conv2_b) +regularization_factor*tf.nn.l2_loss(fc1_b) +regularization_factor*tf.nn.l2_loss(fc2_b) +regularization_factor*tf.nn.l2_loss(fc3_b)
optimizer = tf.train.RMSPropOptimizer(decay=0.98, epsilon=0.1, learning_rate=rate)  #try rmsprop
training_operation = optimizer.minimize(loss_operation)

loss_summary = tf.summary.histogram('My_first_scalar_summary', cross_entropy)
#================ evaluate =====================
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={keep_prob:1.0, x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

#================== train =======================
EPOCHS = 100
BATCH_SIZE = 64

with tf.Session() as sess:
    writer = tf.summary.FileWriter('/tmp/nd', sess.graph)
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train_augmented_norm[offset:end], y_train_augmented[offset:end]
            
            sess.run(training_operation, feed_dict={keep_prob:0.75, x: batch_x, y: batch_y})
            
        training = sess.run(loss_summary, feed_dict={keep_prob:0.75, x: batch_x, y: batch_y})
        writer.add_summary(training, i)    
        validation_accuracy = evaluate(X_valid_norm, y_valid)
        
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")