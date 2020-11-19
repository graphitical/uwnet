from uwnet import *
def conv_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 2),
            make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 8, 3, 2),
            make_convolutional_layer(8, 8, 8, 16, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 16, 3, 2),
            make_convolutional_layer(4, 4, 16, 32, 3, 1),
            make_activation_layer(RELU),
            make_connected_layer(512, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)

def my_conv_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 2),
            make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 8, 3, 2),
            make_batchnorm_layer(8),
            make_convolutional_layer(8, 8, 8, 16, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 16, 3, 2),
            make_batchnorm_layer(16),
            make_convolutional_layer(4, 4, 16, 32, 3, 1),
            make_activation_layer(RELU),
            make_connected_layer(512, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)

def my_conv_net2():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 2),
            make_batchnorm_layer(8),
            make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 8, 3, 2),
            make_convolutional_layer(8, 8, 8, 16, 3, 1),
            make_batchnorm_layer(16),
            make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 16, 3, 2),
            make_convolutional_layer(4, 4, 16, 32, 3, 1),
            make_batchnorm_layer(32),
            make_activation_layer(RELU),
            make_connected_layer(512, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)


print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 500
rate = .1
momentum = .9
decay = .005

# m = conv_net()
# m = my_conv_net()
m = my_conv_net2()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# 7.6 Question: What do you notice about training the convnet with/without batch normalization? How does it affect convergence? How does it affect what magnitude of learning rate you can use? Write down any observations from your experiments:
# TODO: Your answer

'''
I added 2 batch norm layers after each maxpool layer.
I know the original paper suggested batch norm before activation, but it seems the prevailing thought is to do it after the activation.
I didn't find anything about doing it before or after max pool though so I did it after.

I was able to come to a better final accuracy using batch norm. 
Without batchnorm accuracy ~= 40%. 
With batchnorm accuracy ~= 55%.

At the same learning rate (L = 0.01) I didn't notice a significant difference in convergence.
I then tried L = 0.1 (like the github recommended) and it seemed to converge more quickly (~200 iterations), but I ended with a worse accuracy ~51%

I then tried without batch norm and L = 0.1 and noticed it took significantly longer to converge and actually never reached the best accuracy seen. I got ~37%

So it is clear that batch norm allows us to use a larger learning rate, which is helpful.
But we still need a smaller learning rate to get the highest final accuracy.

FINALLY: I just reread the github page and it recommends trying batch norm immediately after the convolutional layers. This gives me 3 batch norms now.
This actually seems to converge worse than the batch norm after maxpool and has a worse final accuracy of ~51%

CONCLUSIONS:
Best way was to have 2 batch norm layers after each of the maxpools.
This gave me the highest accuracy and fastest convergence.
Using batch norm always allowed for a higher learning rate and a better convergence over not using it though.
'''