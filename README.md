# Restricted Boltzmann Machine
implement rbm using both Boost and Armadillo library in C++

# Author
Il Gu Yi


# Usage
- execute: ./rbm train.d paras.rbm.d
- argument 1: train data
- argument 2: parameters data
- DATA: MNIST (partial data) 


# What program can do (version 0.1) (2015. 07. 17.)
- both Binary RBM and Gaussian Bernoulli RBM
- constraint standard deviations are 1 in GBRBM case



# Requirement
- I use the random number generator mt19937 from Boost library
for weights and bias initialization and stochastic gradient descent.
- I implement my program using Armadillo linear algebra library in C++
for various calculation based on matrix and vector.



