import mnist_loader
import network2

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
validation_data = list(validation_data)

#training_data = training_data[0:1000]
net = network2.Network([784,30,10])
net.SGD(training_data, 30, 10, 0.1, lmbda=5, evaluation_data = validation_data, monitor_training_accuracy=True, monitor_evaluation_accuracy = True)
