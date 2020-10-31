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

'''
https://www.machinecurve.com/index.php/2020/02/09/how-to-build-a-convnet-for-cifar-10-and-cifar-100-classification-with-keras/
# Create the model
model = Sequential()
Xmodel.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
Xmodel.add(MaxPooling2D(pool_size=(2, 2)))
Xmodel.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
Xmodel.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(no_classes, activation='softmax'))
'''

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
    l = [   make_connected_layer(32 * 32 * 3, 1536),
            make_activation_layer(RELU),
            make_connected_layer(1536, 768),
            make_activation_layer(RELU),
            make_connected_layer(768, 384),
            make_activation_layer(RELU),
            make_connected_layer(384, 128),
            make_activation_layer(RELU),
            make_connected_layer(128, 10),
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
# batch = 128
batch = 1
iters = 5000
rate = .01
momentum = .9
decay = .005

# m = conv_net()
# m = my_conv_net()
# m = my_fc_net()
m = debug_conv_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# How accurate is the fully connected network vs the convnet when they use similar number of operations?
# Why are you seeing these results? Speculate based on the information you've gathered and what you know about DL and ML.
# Your answer:
#

