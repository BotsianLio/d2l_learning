batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 10)
                   )

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);
#nn.sequential: module that allows you to sequentially stack neural network layers
#nn.Flatten: a module that is used to flatten multi-dimensional input tensors into a one-dimensional tensor

loss = nn.CrossEntropyLoss(reduction='none')
#LogSumExp

trainer = torch.optim.SGD(net.parameters(), lr=0.1)
#END

num_epochs = 10
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
