from uwnet import *

def conv_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(32, 32, 8, 3, 2),
            make_convolutional_layer(16, 16, 8, 16, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 16, 3, 2),
            make_convolutional_layer(8, 8, 16, 32, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 32, 3, 2),
            make_convolutional_layer(4, 4, 32, 64, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(4, 4, 64, 3, 2),
            make_connected_layer(256, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)

# Based on
# https://www.machinecurve.com/index.php/2020/02/09/how-to-build-a-convnet-for-cifar-10-and-cifar-100-classification-with-keras/
def my_conv_net():
    l = [   make_convolutional_layer(32, 32, 3, 32, 3, 1), make_activation_layer(RELU),
            make_maxpool_layer(32, 32, 32, 2, 2),
            make_convolutional_layer(16, 16, 32, 64, 3, 1), make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 64, 2, 2),
            make_convolutional_layer(8, 8, 64, 128, 3, 1), make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 128, 2, 2),
            make_connected_layer(2048, 256),
            make_connected_layer(256, 128),
            make_connected_layer(128, 10),
            make_activation_layer(SOFTMAX)
            ]
    return make_net(l)

def my_fc_net():
    l = [   make_connected_layer(32 * 32 * 3, 2048),
            make_activation_layer(RELU),
            make_connected_layer(2048, 1024),
            make_activation_layer(RELU),
            make_connected_layer(1024, 512),
            make_activation_layer(RELU),
            make_connected_layer(512, 256),
            make_activation_layer(RELU),
            make_connected_layer(256, 10),
            make_activation_layer(SOFTMAX)
            ]
    return make_net(l)

def debug_conv_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 2), 
            make_activation_layer(RELU),
            make_convolutional_layer(16, 16, 8, 16, 3, 2), 
            make_activation_layer(RELU),
            make_convolutional_layer(8, 8, 16, 32, 3, 2), 
            make_activation_layer(RELU),
            make_convolutional_layer(4, 4, 32, 64, 3, 2), 
            make_activation_layer(RELU),
            make_connected_layer(256, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)

print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
# batch = 2
iters = 5000
rate = .01
momentum = .9
decay = .005

# m = conv_net()
m = my_conv_net()
# m = my_fc_net()
# m = debug_conv_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# Default conv_net: Training Accuracy: 69%, Test Accuracy: 64%
# My conv net: Training Accuracy: 85%, Test Accuracy: 71%
# FC net: Training Accuracy: 60%, Test Accuracy: 53%

# How accurate is the fully connected network vs the convnet when they use similar number of operations?
# Why are you seeing these results? Speculate based on the information you've gathered and what you know about DL and ML.
'''
The FC network had 60% training accuracy and 53% test accuracy indicating some overfitting. The default conv_net has
69% training accuracy and 64% test accuracy indicating some overfitting, but not as much as the FC network. Both have
a similar number of FLOPs in them (ignoring maxpools and bias addition).

Why this difference? Obviously the improved performance is in the network architecture, not just more calculations. I think
this has to do with the fact that a CNN focuses on learning more about neighborhoods rather than global connections. In an
image the pixels nearest to each other have a higher likelihood of being correlated, while pixels further away will tend to
be less correlated. This makes intuitive sense and makes use of the convolution (cross-correlation) process to extract this
local info. A fully connected network makes no distinction between pixel location and instead feeds all pixels indescriminately 
forward and assuming all should be correlated. 

I think this means that neurons cannot extract as much fine grained information and instead of specializing become more generalized,
leading to lower overall accuracy. The disparity in overfitting I don't have quite an intuitive grasp on. Ultimately overfitting
means the model is extracting too much information from the training set. I also know that a fully connected network with have greater
variance in the network than with a conv net. I think the overfitting of the fully connected layer is worse than the conv net because
the fully connected network is a "wider net" that regresses to the mean more than a more specialized conv net.

I don't know I could be totally making that up though...




### Calculating FLOPs ###
EDIT: I missed the part about FMA and just figured this out myself. This doesn't affect the number
of neurons because both equations are off by the same constant (2) which cancels out in the algebra.
It would artificially increase the number of expected operations though by a factor of 2.

In general, for multiplying matrices A*B that are of size (l x m) * (m x n) 
you have a repeated dot product operation down all row vector vectors of A and 
across all columns of B. For a single row vector/column vector pair this results 
in m multiplications and m-1 additions. You repeat this procedure for all 
rows down A (l) and all columns across B (n). For our first layer, l = 8, m = (32*32*3), & n = 32^2.
conv_FLOPs = (m + m - 1)  l  n ~= 2*m*l*n ~= 25MFLOPs

By comparison the matrix multiplication for a fully connected layer of x 
the image (1 x m) and W the weight matrix (m x p) where p is the number 
of neurons is (1 x m) * (m * p). Ignoring bias addition,z activation, etc. 
with sizes (1 x m) * (m x p) yields:
fc_FLOPs ~= 2*m*p

Setting these equal and solving for p (to get the number of neurons, which is what we want)
gives p = l*n. In our case p = 8 * (32^2) = 2^13 (for stride 1).

In our case accouting for the maxpool stride we get 4 fully connected layers with [2048, 1025, 512, 256]
neurons respectively because we have [110592, 147456, 147456, 147456] operations per layer respectively.

Thanks for coming to my TED talk.
'''