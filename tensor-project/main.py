import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # y labels are oh-encoded

#  image data is stored into list using one hot encoding.

#  encoding represents abstract (from a computer's perspective) information
#  as a datastructure populated with values that represent the information.

#  encoding allows a computer to understand information that it otherwise wouldn't

#  one-hot encoding uses a vector of any dimension to represent data with binary values

#  in this example, the encoding is a single-dimension vector with 
#  1's and 0's representing pixel colors in a 28x28 image.
#  a 1 represents a black pixel and a 0 represents a white pixeli

n_train = mnist.train.num_examples  # 55,000 population size
n_validation = mnist.validation.num_examples  # 5,000 population size
n_test = mnist.test.num_examples  # 10,000 population size


#  this is the part where the layers of the neural network are written
#  neural networks are inspired by the human brain, and their methods
#  can be imagined as the receival, modification, and input of information
#  between each layer. The n_input layer takes in the encoded information,
#  the hidden layers to modify the values, and the output layer to present
#  the information

#  in this example, the values assigned to each layer represent ____________
#  there are 784 inputs represented by the area of the image (28x28 pixels)


n_input = 784  # input layer that recieves the encoding (28x28 pixel image)
n_hidden1 = 512  # 1st hidden layer
n_hidden2 = 256  # 2nd hidden layer
n_hidden3 = 128  # 3rd hidden layer
n_output = 10  # output layer (0-9 digits)

#  a neural network has certain constant values called 'hyperparameters' 
#  these values remain fixed for the course of the training
#  modifying these values is the best way to optimize the NN

learning_rate = 1e-4
n_iterations = 1000
batch_size = 128
dropout = 0.5

#  now its time to build the NN using TensorFlow
#  the backbone of the module is a tensor
#  its a datastructure like an array or a list
#  now making 3 tensors; an input, an output, and a controller

X = tf.placeholder("float", [None, n_input])  # 784 possible locational inputs with an unknown number of images
Y = tf.placeholder("float", [None, n_output])  # 10 possible classes of output with an unknown number of outputs
keep_prob = tf.placeholder(tf.float32)  # controls the dropout rate 

#  in the training process the NN will change the weight and bias values, so they need an initial placeholder
#  in these values the network does its learning, as they represent the strength of connection between units
#  their initial values are impactful on the final accuracy of the NN, so we want them close but not equal to 0

weights = {
        'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden1], stddev=0.1)),
        'w2': tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2], stddev=0.1)),
        'w3': tf.Variable(tf.truncated_normal([n_hidden2, n_hidden3], stddev=0.1)),
        'out': tf.Variable(tf.truncated_normal([n_hidden3, n_output], stddev=0.1)),
        }

#  in the bias a small constant value is used to ensure tensor activation & contribution to propogation

biases = {
        'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden1])),
        'b2': tf.Variable(tf.constant(0.1, shape=[n_hidden2])),
        'b3': tf.Variable(tf.constant(0.1, shape=[n_hidden3])),
        'out': tf.Variable(tf.constant(0.1, shape=[n_output]))
        }

#  building layers & writing the code to manipulate the tensors

layer_1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])
layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
layer_3 = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
layer_drop = tf.nn.dropout(layer_3, keep_prob)
output_layer = tf.matmul(layer_3, weights['out']) + biases['out']

#  each hidden layer does a matrix multiplication with the previous layer's output using the current layer's weights    
#  then adds the biases to the values

#  now the loss function will be defined. The chosen function is the cross-entropy loss function, aka log-loss
#  this function is popular with TensorFlow applications.
#  the function quantifies the differnece between two probability distributions
#  in this example those distributions would be the predictions and the labels
#  a perfect function would have a cross-entropy of 0, aka no loss

#  another function that will be needed is an optimization algorithm which will be used to minimize the loss function
#  a common process for this is the gradient descent optimization, and the variation being used is the Adam optimizer.
#  this function finds the local minimum lby taking iterative steps along the gradient in a descending direction

cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=Y, logits=output_layer
            ))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#  the ai code is complete. now its time to run a lot of tests on it

#  here is some code to display the AI's progress as it trains
#  arg_max compes which images are being predicted correctly by looking at the output_layer and Y variable
#  by using tf.equal a list of bools is returned and then casted to floats. the mean of this list is the accuracy score

correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



#  inputting data to train on
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#  overall, this code can be described as the continuous optimization of an arbitrary loss function
#  this loss function outputs the dissimilarity score between the handwritten number and the number guessed by the NN
#  to optimize this loss function means the AI's results are better

#  the process of training:
#  -> Propagate values forward through the network
#  -> Compute the loss
#  -> Propagate values backward through the network
#  -> Update the parameters

# train on mini batches
for i in range(n_iterations):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={
        X: batch_x, Y: batch_y, keep_prob: dropout
        })
    # print loss and accuracy (per minibatch)
    if i % 100 == 0:
        minibatch_loss, minibatch_accuracy = sess.run(
                [cross_entropy, accuracy],
                feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0}
                )
        print(
                "Iteration",
                str(i),
                "\t| Loss =",
                str(minibatch_loss),
                "\t| Accuracy =",
                str(minibatch_accuracy)
                )

#  note: loss & accuracy data per batch should be similar

test_accuracy = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0})
print("\nAccuracy on test set:", test_accuracy)
