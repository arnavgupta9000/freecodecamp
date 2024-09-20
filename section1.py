#https://www.youtube.com/watch?v=V_xro1bcAuA

#colab.research.google.com
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss
'''
scalar = torch.tensor(7)
print(scalar)

print(scalar.ndim)

# get tensor back as python int
print(scalar.item())
'''


'''
vector = torch.tensor([7,7]) # number of square brackets = dimension size
print(vector)
print(vector.ndim) # 1
print(vector.shape) # 2 -> 2 by 1 elements = total of 2 elements = [2]
'''

'''
matrix = torch.tensor([[7,8], [9,10]])
print(matrix)
print(matrix.ndim) # 2
print(matrix[0]) # [7,8]
print(matrix.shape) # 2 cols 2 rows = [2,2]
'''
'''
tensor = torch.tensor([[[1,2,3], [3,6,9], [2,5,4]]])
print(tensor)
print(tensor.ndim) # 3
print(tensor.shape) # [1,3,3] -> outer, rows, # of elements per row

'''

'''
tensor = torch.tensor([[2,2,2,3], [1,1,1,1]]) # all rows have to be the same size (4 in this case)
print(tensor, tensor.ndim, tensor.shape, sep = '\n')

'''
'''
# random tensors -> why? random tensors are important because the way many neural networks learn is that they start with tensors full of random numbers and the adjust those random numbers to better represent the data

# start with random numbers -> look at data -> uptade random numbers -> look at data -> update random numbers

# create reandom tensor size(3,4)

random_tesnor = torch.rand(3,4) ## (1,3,4) as many args as we need
print(random_tesnor, random_tesnor.ndim, sep = '\n')

# create a radnom tensor with similar shape to an image tensor
image = torch.rand(size=(224,224,3)) # height, width, color channels (R, G, B) you dont need size but can include it (or do it without size) image = torch.rand(224,224,3)
print(image, image.shape, image.ndim, sep = '\n')
# size = dimensions, ndim = number of layers, tensor itself = print the tensor itself


# create a tensor of all zeros
zero = torch.zeros(3,4)
print(zero)
print(random_tesnor * zero) # used for block certain rows and columns aka masking

# tensor of all ones
ones = torch.ones(3,4)
print(ones, ones.dtype, sep='\n') # float32 default for torch
'''

'''
# creating range of tensors and tensors like
num = torch.arange(1,11) # start end step like for loop
print(num)

# creating tensors like
zero = torch.zeros_like(num) # same dimensions - torch.ones_like(num) also works
print(zero)
'''
'''
# float_32 = torch.tensor([3.0,6.0,9.0], dtype=torch.float16)
# print(float_32.dtype) # float16


float_32 = torch.tensor([3.0,6.0,9.0], 
                        dtype=None, # what datatype is the tensor (e.g float32 or float16)
                        device = None, # what device your tensor is on
                        requires_grad= False) # weather or not to track gradients with this tensors opperation
print(float_32.dtype) # float32 still

float_16_tensor = float_32.type(torch.float16) # or the other way above
print(float_16_tensor.dtype)
print(float_16_tensor * float_32) ## acc works?

'''

'''
int_32_tensor = torch.tensor([3,6,9], dtype=torch.int32)
float_32 = torch.tensor([3.0,6.0,9.0])
print(float_32 * int_32_tensor) ## still works

## getting information from tensors
# tensors not right datatype - to do get datatype from a tensor, can use `tensor.dtype`
# tensors not right shape - to get shape from a tensor, can use `tensor.shape`
# tensors not on the right device - to get device from a tensor, can use  `tensor.device`

some_tensor = torch.rand(3,4)
print(some_tensor)
# find out details about tensor
print(f"Datatype of tensor: {some_tensor.dtype}\nShape of tensor: {some_tensor.size()}\nTensor device is on: {some_tensor.device}")
#tensor.size() == tensor.shape i would use the second one
'''

'''
# maniuplating tensors (tensor operations)
# tensor operations include - addition, subtraction, multiplication (element wise), division, matrix multiplication

tensor = torch.tensor([1,2,3])
print(tensor + 10) # 11,12,13 BUT its not saved ie after this line tensor = [1,2,3]
tensor *= 10 
print(tensor)
tensor = torch.tensor([1,2,3])
print(tensor - 10)
print(torch.mul(tensor, 10)) # 10, 20,30
print(tensor) # [1,2,3] result not saved
print(torch.add(tensor,10))
''' 

'''
## matrix multiplication

# two main ways of performing multiplication in neural networks and deep learning
#1.element wise multiplication
#2. matrix multiplication (dot product)

tensor = torch.tensor([1,2,3])
print(tensor * tensor) # 1 4 9

#matrix multiplication
print(torch.matmul(tensor, tensor)) # 14 -> normal matrix multiplication
#print(tensor @ tensor) does the same thing but prob use ^^
# 2 main rules for this to work
# 1. inner dimensions must match
# (3,2) @ (2,1) = (3,1)
'''

'''
## one of the most common issues in deep learning is shape error 

tensor_A = torch.tensor([[1,2], [3,4], [5,6]])
tensor_B = torch.tensor([[7,10], [8,11], [9,12]])

# print(torch.mm(tensor_A, tensor_B)) ## torch.mm == torch.matmul (same thing)

# to fix our tensor shape issues, we can manipulate the shape of one of our tensors using a **transpose**
# transpose switches dimensions
print(tensor_B.T, tensor_B, sep = '\n')
## .T = transpose

print(torch.mm(tensor_A, tensor_B.T)) ## works now 

'''

'''
## finding the min, max, mean, sum etc (tensor aggregation)

# tensor = torch.rand(3,4)
tensor = torch.arange(1,100,10)
#tensor = torch.arange(0,100,10, dtype = torch.float32) this also works here
print(torch.min(tensor), tensor.min()) # both work here
print(torch.max(tensor), tensor.max())
print(torch.mean(tensor.type(torch.float32)), tensor.type(torch.float32).mean()) # the arange tensor doesnt work here without changing the dtype since it wont accept type `long` ie torch.mean() requires flaot32
print(torch.sum(tensor), tensor.sum())
# i would use torch.function(tensor)

# find the pos in tensor that has the min value with argmin() -> returns index position of target tesnor where the min value occurs
print(torch.argmin(tensor))
# same thing with max
print(torch.argmax(tensor))

'''
'''
## reshaping, stacking, squeezing, and unsqueezing tensors

# reshaping - reshapes and input tensor to a defined shape
# view - return a view of an input tensor of certain shape but keep the same memory as the original owner
# stacking - combine multiple tensors on top of each other (vstack) or side by side (hstack)
# squeeze - removes all `1` dimensions from a tensor
# unsqueze - add a `1` dimension to a target tensor
# Permute - return a view of the input with dimensions permuted(swapped) in a certain way

x = torch.arange(1.,10.) # the '1.' makes it into a float 
# print(x, x.dtype, sep = '\n') #tensor([1., 2., 3., 4., 5., 6., 7., 8., 9.]), torch.float32

print(x,x.shape) # size([9])
# x_reshaped = x.reshape(1,7)
# print(x_reshaped, x_reshaped.shape) # error 9 elements cannot be squeezed to 7 dimensions
x_reshaped = x.reshape(1,9) # basically adds a an extra  -> the orgingal shape(z) should be equal to (x,y) such that x * y = z. also the layer of brackets notice how its 1,9. it becomes tensor[[1], [2]]... if it was (9,1) then it would be [[1,2...]]
# notice the reshape works like (well with 2 numbers (x,y)) x rows of y elements in each row
print(x_reshaped, x_reshaped.shape) 

z = x.view(1,9)
print(z, z.shape)
#changing z changes x - view of a tensor shares the same memory as the original input
z[:,0] = 5 # z[0,0]
print(z,x)

# stack tensors on top of each other
x_stacked = torch.stack([x,x,x,x], dim = 0) # dim is a confusing topic
print(x_stacked)

# learn vstack and hstack on our own...


#torch.squeeze() - removes all single dimensions from a  target 
print(x_reshaped, x_reshaped.squeeze()) # removed one square bracket
print(x_reshaped.squeeze().shape) # before squeeze it was [1,9] but now its [9]
x_squeezed = x_reshaped.squeeze()

# torch.unsqueeze() - add a single dimension to a target tensor at a specific dim (dimension)
print(f"previous target {x_squeezed}")
print(f"prev shape {x_squeezed.shape}")
x_unsqueezed = x_squeezed.unsqueeze(dim = 0) # adding an extra dimension with unsqueeze at pos 0 change it to dim=1 to see difference

print(f"new tensor {x_unsqueezed}, \n New shape {x_unsqueezed.shape}")

# torch.permute - rearranges the dimensions of a target tensor in a specified order - returns a view - shares memory remember - the params are the order of the DIMENSIONS

x_original = torch.rand(224,224,3) # [height, width, color channels]
# permute the og tensor to rearrange the axis(or dim) order
x_permuted = x_original.permute(2,0,1) #[color channel, height, width], ie shifts 0->1, 1->2, 2->0
print(f"prev shape {x_original.shape}\nNew shape {x_permuted.shape}") 
'''
'''
## Indexing (selecting data from tensors)

## indexing with pytorch is similar to indexing with numpy
x = torch.arange(1,10).reshape(1,3,3) #-> 9 = 1 * 3 * 3
print(x, x.shape)
# lets index on our new tensor

print(x[0])

# lets index on the middle bracket (dim = 1)
print(x[0][0]) #x[0,0] also works
# lets index on the most inner bracket (last dimension)
print(x[0][0][0]) # 1

# u can use ":" to select "all" of a target dimension
#print(x[:0]) # tensor([], size=(0, 3, 3), dtype=torch.int64)
print(x[:, 0]) # [1,2,3]
# get all values of 0th and first dimensions but only index 1 of 2nd dimensions

print(x[:,:,1])

# get all values of the 0 dimensions but only the 1 index value of the 1st and 2nd dimension
print(x[:, 1, 1]) #tensor([5])
# print(x[0, 1, 1]) #tensor(5) - notice no outer bracket now

# get index 0 of 0th and 1st dimension and all values of 2nd dimension
print(x[0,0,:])

# index on x to return 9
print(x[:, 2, 2])

#index on x to return 3,6,9
print(x[:,:,2])

'''

'''
# pytorch tensors and numpy
# data in numpy, want in pytorch tensor -> `torch.from_numpy(ndarray)`
#pytorch tensor -> numpy -> `tensor.Tensor.numpy()`

array = np.arange(1.0,8.0)
tensor = torch.from_numpy(array).type(torch.float32)
print(array, tensor)
# from numpy to tensor -> float64 but with the float 32 thing we have above it fixes it

array = array + 1
print(array,tensor) # array is changed but tensor is not

#tensor to numpy array
tensor = torch.ones(7)
numpy_tensor = tensor.numpy()
print(numpy_tensor, tensor) # numpy_tensor.dtype = float32 since thats the default for torch

#change the tensor what happens to the numpy_tensor
tensor = tensor + 1
print(numpy_tensor, tensor) # numpy_tensor is unchanged
'''

'''
# reproducibility (trying to take the random out of random)
# in short how a nueral network learns:
# start with random numbers -> perfrom tensor operations -> update random numbers to try and make them better representation of the data -> again -> again ->again...

# to reduce the randomness in neural networks and pytorch comes the concept of a **random seed**
# essentially what the random seed does is "flavour" the randomness

random_tensor_A = torch.rand(3,4)
random_tensor_B = torch.rand(3,4)
print(random_tensor_A, random_tensor_B)
print(random_tensor_A == random_tensor_B) # compares each element not like normal python

# lets make some random but reproducible tensors

random_seed = 42
torch.manual_seed(random_seed)
random_tensor_C = torch.rand(3,4)
torch.manual_seed(random_seed) ## need this here otherwise it wont work... manual seed only works for one `block` of code -> basically uses torch.manual_seed everytime calling random function
random_tensor_D = torch.rand(3,4)
print(random_tensor_C, random_tensor_D, random_tensor_C == random_tensor_D, sep = '\n')

'''

'''
# running tensors and pytorch objects on GPUs (and making faster computations)
#GPUs = faster computation on numbers

# getting a gpu
#1. easiest - use gogle colab for a free gpu
#2. use your own gpu - takes a little bit of setup - reqs a gpu
#3. use cloud computing - GCP, AWS, Azure, these services allow you to rent computers on the cloud and access them

## check for gpu access with pytorch

print(torch.cuda.is_available())

# setup device agnostic code

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# count the number of devices
print(torch.cuda.device_count())


#putting tensors and models on gpu
# https://colab.research.google.com/drive/1G_gYJYYxZkIgZOE3TRsN92vO571tIRzG#scrollTo=zpd_WVOsv4u_
'''