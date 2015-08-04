#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <ctime>
#include <boost/random.hpp>
#include <armadillo>
#include "dataframe.h"
#include "rbm.h"
using namespace std;
using namespace df;
using namespace rbm;



int main (int argc, char** argv)
{
    if ( argc != 3 ) {
        cout << "Usage: rbm_online train_data parameters_data" << endl;
        exit(1);
    }

    //  Reading Parameters file
    BinaryRBM rbm;
//    GaussianBernoulliRBM rbm;
    rbm.ReadParameters(argv[2]);
    rbm.PrintParameters();

    unsigned N = rbm.GetN_train();
    unsigned dimension = rbm.GetDimension();


    //  Reading Input data file
    DataFrame<unsigned> trainMNIST;
//    Usage: ReadDataFile(filename, N, dimension, header, target)
    trainMNIST.ReadDataFile(argv[1], N, dimension, "True", "True");
    cout << "Read complete" << endl;
//    trainMNIST.PrintData();


    //    Transform Binary Data for Binary Restricted Boltzmann Machines
    trainMNIST.TransformBinaryData();
    cout << "Transform binary data complete" << endl;


    //    Set linear scaling each features for Gaussian Bernoulli Restricted Boltzmann Machines
//    DataFrame<double> trainMNIST_norm;
//    trainMNIST.LinearScalingEachFeatures(trainMNIST_norm);
//    trainMNIST.NormalizationEachFeatures(trainMNIST_norm);
//    cout << "Normalize training data complete" << endl;
//    trainMNIST_norm.PrintData();



//  Execute Restricted Boltzmann Machines
    rbm.Initialize("uniform");

    for (unsigned step=0; step<1; step++) {
        rbm.Training(trainMNIST, step);
//        rbm.Training(trainMNIST_norm, step);
    }




    return 0;
}



