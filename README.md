
# MNIST Machine Learning

A Machine Learning model trained from the MNIST Database to distinguish hand-written numbers. 
This is a very simple example of a Machine Learning Algorithm and I have coded this to help get a better understanding of the fundamental maths for Machine Learning.


## Interesting Guides

 - [Extremely Helpful Guide - Samson Zhang](https://awesomeopensource.com/project/elangosundar/awesome-README-templates)
- [The Detailed Documentation to his Project](https://www.kaggle.com/code/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras)
 

## Initialization

Install the required dependencies

```bash
  pip install -r requirements.txt
```
    
    
# The Explanation 

The main reason for this project was to understand the maths behind a machine learning algorithm. 
I was heavily incentivized by the rapid growth of AI and how integrated it has become in our day-to-day life.

The best way I thought to learn this was to avoid using any specialised libraries, such as PyTorch, TensorFlow, etc and only use libraries that would help with integrating the fundamental maths. I ended up only using 3 external libraries:

- [Pandas (To read CSV Data)](https://pandas.pydata.org/docs/)
- [NumPy (Used for creating equations)](https://numpy.org/doc/)
- [MatPlotLib (For visualising the final predictions)](https://matplotlib.org/stable/index.html) 

I pulled the MNIST Dataset from Kaggle.


**//**

This explanation will not explain how to integrate the algorithms as Python Code (A guide to do so has already been linked) , but focus solely on the maths and what each algorithm does and how everything relates.

**//**
## What is Machine Learning?

A machine learning model is a program that learns patterns from data to make predictions or decisions. It adjusts its internal parameters based on patterns, and then applies what it has learned to new, unseen data.

A simple model ML Model works as shown in the diagram below:

![](https://imgur.com/jGkzEvI.png)

- The Input Layer refers to the very first data that **recieves the data**. This is before any processing occurs.

- The Hidden Layer is where the neural network learns to **extract patterns** from the raw input data. This is where all the maths happens and is the most important section of a ML Model.

- The Output Layer takes the data from after processing and **applies a softmax function** to convert the data into a probability from 0 to 1.

In context to this project, each number is a 24x24 grid containing a handwritten number.
This corresponds to **784 Pixels**. The different choices for the model will be between 0 and 9, giving **10 potential options** for predictions

To allow for easier manipulation and visualization, data will be stored as matrices.

- The **Input Layer** will be a matrix of (784 x 1)
- The **Hidden Layer** will be a matrix of (10 x 1)
- The **Output Layer** will be a matrix of (10 x 1)

## Hidden Layer

A simple equation to model this layer is: 

![](https://imgur.com/4yEhgOM.png)

Where Z represents the **Hidden Layer**, W is the **Weight**, X is the **input** and b is the **Bias**

The Weight and the Bias in the equation are both learnt values that change with each training iteration. These values change by utilising Backward Propogation and ReLU functions

### ReLU Function
A ReLU function adds a layer of non linearity to the algorithm, and forms the basis of the machine learning. It allows for the model to learn complex, non-linear patterns. Without it, neural networks would simply be a series of linear transformations. This is defined as the **Activation Function**, where:

![](https://imgur.com/LYlgyk0.png)

This Activation function is very simple compared to others, (sigmoid, tanh)

### Backward Propogation

Backward Propogation is an algorithm used to compute gradients from loss functions with respect to the weights used in the neural network. This simply put, calculates how much each weight contributes to the error in the predictions. This is then used to update the parameters of the network to make the predictions more accurate. The backward propogation algorithm is:

![](https://imgur.com/13l3MxW.png)

Where A is the **Activation Layer**, m is the **Sample Size** and Y is the the **Correct Prediction**

This works by calculating the gradients at each layer and for each separate bias. An average value is taken from each weight and bias. 

At dZ[2] the model is calculating how far the prediction is from the correct prediction. 
At dz[1], The model works backwards from the next layer using the weights to figure out how much it contributed to the error. 

With these error values calculated, it is then removed from the initial bias and weights that were used, modelled as:

![](https://imgur.com/OqDK3dE.png)

Where 𝛼 is the **Learning Factor**. This is the only value which is inputted by the user and controls how much the algorithm will modify values by with each iteration. This can be fine tuned to find a value which can predict with the greatest accuracy, however it can also cause the model to be less efficient.

### One Hot Y Function
This is what the **Y** value in the backpropogation is. This initialises a (10x1) Numpy Matrix, where each value is a 0. For the value of Y, the function locates that position on the matrix and changes its value from 0 to 1 (eg 1 would be (0,1,0,...0,0)). This is fundamental for backpropogation as without it the error in the last layer would not be computable.

### Forward Propogation

This is where the model uses the weights and biases to generate a prediction. In this case, a 24x24 image of a handwritten number is given, which the model will attempt to calculate what number it is.This is what happens before the Backwards Propogation to help generate the errors. This is modelled by:

![](https://imgur.com/0unRWXk.png)

X is the **input matrix** (784x1) which is used to calculate the dot product between itself and the weight

### Softmax Function
This function is used to change the final values to probabilities between 0 and 1. This works by taking the exponential of Z (Being the raw output scores) and dividing it by the sum of all exponentiated values in that list. This allows for a more legible and user-friendly output, clearly showing the probability. This is modelled as:

![](https://imgur.com/eEGmpWA.png)

After all the calculations, the value in the final matrix with the highest probability is outputted as the final prediction to the user.


## Lessons Learned

This project was extremely eye opening to me for how ML is integrated and the maths was really interesting to learn. Zhang's tutorial was extremely helpful for me as it provided a one stop place for a clear explanation on everything I needed to know, and was very straighforward. For anyone else wanting to learn more about ML, I would highly recommend his guides, which I have linked above.

Although I followed a tutorial in order to program this model, this project has provided me with the fundamental knowledge in order to code my own ML algorithm. 

For my next project, I intend to code a model that can identify different animals from a picture. I will not use any specialised libraries such as PyTorch and TensorFlow; I solely intend to use NumPy. I believe this is a good next step to further my knowledge and understanding, and is a vital skill in the ever growing industry of ML and AI


## Authors

- [@erkanms](https://www.github.com/erkanms)



[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-turquoise.svg)](https://opensource.org/licenses/)


