import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.optim as optim

torch.manual_seed(1) # set the random seed

# for this example, the neural network was changed to have 3 layers, with the same number of hidden units in each later
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 30)
        self.layer2 = nn.Linear(30, 30)
        self.layer3 = nn.Linear(30, 1)
    def forward(self, img):
        flattened = img.view(-1, 28 * 28)
        activation1 = self.layer1(flattened)
        activation1 = F.relu(activation1)
        activation2 = self.layer2(activation1)
        activation2 = F.relu(activation2)
        activation3 = self.layer3(activation2)
        return activation3

def train_network(net):

    # load the data
    mnist_data = datasets.MNIST('data', train=True, download=True)
    mnist_data = list(mnist_data)
    mnist_train = mnist_data[:1000]
    mnist_val   = mnist_data[1000:2000]
    img_to_tensor = transforms.ToTensor()


    # define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

    for (image, label) in mnist_train:
        # actual ground truth
        actual = torch.tensor(label < 3).reshape([1,1]).type(torch.FloatTensor)
        # net prediction
        out = net(img_to_tensor(image))
        # update the parameters based on the loss
        loss = criterion(out, actual)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # computing the error and accuracy on the training set
    error = 0
    for (image, label) in mnist_train:
        prob = torch.sigmoid(net(img_to_tensor(image)))
        if (prob < 0.5 and label < 3) or (prob >= 0.5 and label >= 3):
            error += 1
    print("Training Error Rate:", error/len(mnist_train))
    print("Training Accuracy:", 1 - error/len(mnist_train))

    # computing the error and accuracy on a test set
    error = 0
    for (image, label) in mnist_val:
        prob = torch.sigmoid(net(img_to_tensor(image)))
        if (prob < 0.5 and label < 3) or (prob >= 0.5 and label >= 3):
            error += 1
    print("Test Error Rate:", error/len(mnist_val))
    print("Test Accuracy:", 1 - error/len(mnist_val))

def hyperparameter_results():
    lr = [[0.001, 0.922, 0.887], [0.005, 0.964, 0.921], [0.010, 0.961, 0.918], [0.050, 0.688, 0.703]]

    print("Learning Rate|Training Accuracy|Test Accuracy\n---------------------------------------------")
    for i in lr:
        print(' ', str(i[0]).ljust(5, '0'), '     |     ', i[1], '     |     ', i[2])

    layers = [[2, 0.964, 0.921], [3, 0.955, 0.921], [4, 0.959, 0.907]]


    print("Layers|Training Accuracy|Test Accuracy\n--------------------------------------")
    for i in layers:
        print(' ', i[0], '  |     ', i[1], '     |    ', i[2])

    act_func = [['relu', 0.964, 0.921], ['leaky_relu', 0.963, 0.921], ['tanh', 0.960, 0.906]]


    print("Activation Function|Training Accuracy|Test Accuracy\n--------------------------------------------------")
    for i in act_func:
        print(' ', i[0].ljust(10, ' '), '      |     ', str(i[1]).ljust(5, '0'), '     |   ', i[2])

if __name__=='__main__':
    net = Network()
    train_network(net)
    hyperparameter_results()