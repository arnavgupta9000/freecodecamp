#https://www.youtube.com/watch?v=V_xro1bcAuA

import torch
from torch import nn as nn # nn contains all of pytorch's building blocks for neural networks
import numpy as np
import matplotlib.pyplot as plt
'''
what_were_covering = {1:"data(prepare and load)",
                      2: " build model",
                      3: "fitting the model to data (training)",
                      4: "making predicitions and evaluting a model (inference)",
                      5: "putting it all together"}


# print(torch.__version__)

## 1. Data (preparing and loading)
# data can be almost anything... in machine learning
# excel spreadsheet, images of any kind, videos, audio like songs or podcast, DNA, text
# machine learning is a game of two parts
# 1. Get data into a numerical representation
# 2. Build a model to learn patterns in that numerical representation

# to showcase this, let's create some *known* data using the linear regression formula -> y= mx + b
# we'll use a linear regression formula to make a straight line with know **parameters**

# create *known* parameters
weight = 0.7 # m
bias = 0.3 # b from mx + b

start = 0
end = 1
step = 0.02
x = torch.arange(start, end, step).unsqueeze(dim = 1) # adds the extra dimensions ie we get more square brackets
y = weight * x + bias

# print(x[:10], y[:10], len(x), len(y), sep = '\n')



# splitting data into training and tests sets (one of the most important concepts in machine learning in general)

# lets create a training and test set with out data
# create a train/test split

train_split = int(0.8 * len(x))
x_train, y_train =  x[:train_split], y[:train_split]
x_test, y_test = x[train_split:], y[train_split:]
# print(len(x_train), len(y_train), len(x_test), len(y_test))


# how might we better visualize our data?

def plot_predictions(train_data = x_train, train_labels = y_train, test_data = x_test, test_labels = y_test, predictions = None):
   
    # Plots training data, test data, and compares predictions
    
    plt.figure(figsize=(10,7))

    # plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # plot test data in green
    plt.scatter(test_data, test_labels, c='g', s=4, label = "Testing data")

    # are there predictions?
    if predictions is not None:
        # plot the predictions if they exist
        plt.scatter(test_data, predictions, c='r', s=4, label = "Predictions")
    
    # Show the legend
    plt.legend(prop = {"size": 14})

#plot_predictions()
#plt.show() # need this since we're not in jupitar notebook

# ^^ uncomment those 2 lines to see the graph

# 2. build model, our first pytorch model

# what our model does:
# start with random values (weight and bias)
# look at training data and adjust the random values to better represent (or get closer to) the ideal values (the weight & bias values we used to create the data)

# how does it do so?
# Through two main algorithms:
# 1. Gradient descent
# 2. Back Propogation

# create a linear model class
class LinearRegressionModel(nn.Module): # <- almost everything in pytorch inherits from nn.Module
    def __init__(self):
        super().__init__()
        #randn = random but following normal distribution whereas rand = [0,1]
        self.weights = nn.Parameter(torch.randn(1, # <- start with a random weight and try to adjust the idea weight
        requires_grad= True,  # <- can this parameter be updated via gradient descent?
        dtype= torch.float)) # <- pytorch loves datatype torch.float32
        # requires_grad = does it need a gradient default is true
        # dtype = just the type 

        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    # forward method to define the computation in the model - need "foward()" if using nn.Module as a subclass
    def forward(self, x: torch.Tensor) -> torch.Tensor: # <- x is the input data
        #This method will be called automatically when you pass data through the model.
        return self.weights * x + self.bias # this is the linear regression formula
    


# Pytorch model building essentials

# toch.nn - containes all of the buildings for computational graphs (a neural nework can be considered a computation graph)

# torch.nn.Parmeter - what parameters should or model try and learn, often a pytorch layer from torch.nn will set these for us

# torch.nn.Module - the base class for all neural network modules, if you sublass it, you should overwrite forward()

# torch.optim - this where the optimziers in pytroch live, they will help with gradient descent

# def foward() - all nn.Module subclasses require you to overwrite forward(), this method defines what happens in the forward computation


## checking the contents of our pytorch model
# now we've created a model, lets see whats inside
# we can check our model parmaters or whats inside our model using `.parameters()`

# create a random seed

torch.manual_seed(42)

# create an instance of the model (this is a subclass of nn.Module) 
model_0 = LinearRegressionModel()
#check out the parameters
# print(list(model_0.parameters()))
#print(*model_0.parameters()) this also works

# List named parameters

# print(model_0.state_dict()) # we get the names now of the parameters


# making prediction using `torch.inference_mode()`

# to check our model's predictive power, let's see how well it predicts `y_test` based on `x_test`
# when we pass data through our model it's going to run it through the `forward()` method

# print(x_test, y_test) 
# ideally in a perfect world it'll take the x_test values and predict the y_test values

# make predictions with model
with torch.inference_mode(): # inference modes disables gradient tracking - saves computation time and memory with large datasets
    y_preds = model_0(x_test)
    # y_preds = model_0(x_test): Passes the x_test values through the model, calling the forward method to make predictions

# print(y_preds)

# y_preds = model_0(x_test)
# print(y_preds)
# this also works but then the graph gets an error

# plot_predictions(predictions=y_preds)
# plt.show()


### 3. Train model

# The whole idea of training is for a model to move from some *unkown* parameters (these may be random) to some known parameters
# or in other words from a poor representation of the dtat to a better representation of the data

# one way to measure how poor or how wrong your models predictions are is to use a loss function.

# Note: loss function may also be called cost function or criterion in different areas. for our case, we're going to refer to it as a loss function

# things we need to train:

 # loss function: a function to measure how wrong your model's prediction are to the ideal outputs, lower is better

 # Optimizer: takes into account the loss of a model and adjusts the mode's parameters (ex weights and bias in our case) to improve the loss function.
 # 
 # and specifically for pytorch we need : 1) a training loop. 2) a test loop 
#  

# check out our models parameters (a parameter is a value that the model sets itself)
print(model_0.state_dict())

# setup a loss function
loss_fn = nn.L1Loss()

# set up an optimizer (stochastic gradient descent - randomly adjust values -> if i increase weights and it reduces loss keep increasing weights)
optimzier = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.01) # lr = learning rate = possibly the most important hyperparameters you can set


### Building a training loop (and a testing loop) in pytorch

# a couple of things we need in a training loop
#0. loop through the data and do...
#1. forward pass (this invovles data moving through our model's `forward()` functions) to make predicitions on data - also called forward propagation
#2. calc the loss (compare forward pass predicitions to ground truth labels)
#3. Optimizer zero grad
#4. loss backward - move backwards through the network to calculate the gradients of each of the parameters of our model with respect to the loss (**backpropopgation**)
#5. Optimizer step - use the optimzier to adjust our models parameters to try and improve the loss (**gradient descent)

# an epoch is one loop through the data...
epochs = 200
# track different values
epoch_count = []
loss_values = []
test_loss_values = []
# 0. Loop through the data... (this is a hyperparameter because we've set this ourselves)

for epoch in range(epochs):
    # set the model to training mode
    model_0.train() # train mode in pytorch sets all parameters that require gradients to require gradients

    #1. forward pass
    y_pred = model_0(x_train)

    #2. calc the loss
    loss = loss_fn(y_pred, y_train) # difference between our predictions and ideal training model 
    # the order is input, target
    #print(loss)

    #3. optimizer zero grade
    optimzier.zero_grad()

    #4. perform back propogation on the loss with respect to the parameters of the model
    loss.backward()

    #5. step the optimizer (perform gradient descent)
    optimzier.step() # by default how the optimizer changes will accumulate through the loop so... we have to zero them above in step 3 for the next iteration of the loop

    model_0.eval() # turns off different settings in the model not needed for evaulation/testing (dropout/batch norm layers)


    with torch.inference_mode(): # turns off gradient tracking & and couple more things behind the scenes
        
        #1. Forward pass 
        test_pred = model_0(x_test)

        #2. calc the loss - notice its TEST data now
        test_loss = loss_fn(test_pred, y_test)
    if epoch % 10 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)
        # print(f"Epoch {epoch} | Loss: {loss} || test loss: {test_loss}")




# print(model_0.state_dict())

# plot the loss curves
# x-axis, y-axis
plt.plot(epoch_count, np.array(torch.tensor(loss_values).numpy()), label = 'Train loss') # converting loss values to numpy since matlib works with numpy
plt.plot(epoch_count, np.array(torch.tensor (test_loss_values).numpy()), label = 'Test loss') # dont need the np.array() here but 
plt.title("training and test loss curve")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
# plt.show()

# as epochs increases the points get closer and closer together
with torch.inference_mode():
    y_preds = model_0(x_test)

plot_predictions(predictions=y_preds)
# plt.show()


## saving the model

# there are three main methods you should know about for saving and loading model in pytorch
#1.`torch.save()` - allows you to save a pytorch object in python's pickle format
#2. `torch.load()` - allows you to load a saved pytorch object
#3. `torch.nn.Module.load_state_dict()` - this allows to load a models saved state dictionary

# saving our pytorch model

from pathlib import Path

# 1. Create models directory

model_path = Path('ML/freecodecamp/models') # main file/subfile/name of new file
model_path.mkdir(parents=True, exist_ok=True)

#2. Create a model save path

model_name = "01_pytorch_workflow_model_0.pth" # pth = pytorch object, pt also works

model_save_path = model_path / model_name

#3. save the model state dict

print(f"saving model to {model_save_path}")
# torch.save(model_0.state_dict(), model_save_path) # saved model now


## loading a pytorch model

# since we saved our model's `state_dict()` rather then the entire model, we'll create a new instance of our model class and load the saved `state_dict()` into that

# to load in a saved state_dict we have to instatiate a new instance of our model class

loaded_model_0 = LinearRegressionModel()

# load the saves state_dict

loaded_model_0.load_state_dict(torch.load(model_save_path)) # file like object/os path like object
print(loaded_model_0.state_dict())

# make some predictions with our loaded model

loaded_model_0.eval()

with torch.inference_mode():
    loaded_model_preds = loaded_model_0(x_test)

# print(loaded_model_preds == y_preds) 
'''

## 6. putting it all together

# see all steps in one place

## 6.1 Data

# create device-agnostic code - this means if we've got access to a GPU, our code will use it (for potentially faster computing). if not gpu is available, the code will default to cpu

device = "cuda" if torch.cuda.is_available() else "cpu"

# create some data useing the linear regression formula of y = weight * X + bias

weight = 0.7
bias = 0.3

# create range values
start = 0
end = 1
step = 0.02

# create x and y (features and labels)
x=torch.arange(start,end,step).unsqueeze(dim=1) # without unsqueeze errors will pop up

y = weight * x + bias
# print(x[:10], y[:10])

# split data
train_split = int(0.8 * len(x))
x_train, y_train = x[:train_split], y[:train_split]
x_test, y_test = x[train_split:], y[train_split:]

# plot the data

def plot_predictions(train_data = x_train, train_labels = y_train, test_data = x_test, test_labels = y_test, predictions = None):
   
    # Plots training data, test data, and compares predictions
    
    plt.figure(figsize=(10,7))

    # plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # plot test data in green
    plt.scatter(test_data, test_labels, c='g', s=4, label = "Testing data")

    # are there predictions?
    if predictions is not None:
        # plot the predictions if they exist
        plt.scatter(test_data, predictions, c='r', s=4, label = "Predictions")
    
    # Show the legend
    plt.legend(prop = {"size": 14})

plot_predictions(x_train, y_train, x_test, y_test)
# plt.show()

## 6.2 building a pytorch linear model

class LinearRegressionMdoelV2(nn.Module):

    def __init__(self):
        super().__init__()

        # use nn.Linear() for creating the model parameters

        self.linear_layer = nn.Linear(1, 1) # we want input of size 1 and output of size 1. (1 value of x -> 1 value of y)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)
    

# set the manual seed
torch.manual_seed(42)
model_1 = LinearRegressionMdoelV2()
print(model_1.state_dict())