import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
#print(training_data)

net = network.Network([784,100,30,10])
net.SGD(training_data, 5, 10, 3.0, test_data)
