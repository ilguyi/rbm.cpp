/***********************************************************
 * Restricted Boltzmann Machines namespace
 * rbm class (parents class)
 * binary_rbm class (child class)
 * gaussian_rbm class (child class)
 *
 * rbm training (NO classification)
 *
 * 2015. 06.
 * modified 2015. 07. 17.
 * by Il Gu Yi
***********************************************************/

#ifndef RESTRICTEDBOLTZMANNMACHINES_H
#define RESTRICTEDBOLTZMANNMACHINES_H

#include <boost/random.hpp>
#include <armadillo>
#include <dataframe.h>
using namespace std;
using namespace df;

namespace rbm {


typedef arma::uvec HiddenNodes;
typedef arma::mat Weights;
typedef arma::vec Bias;
typedef arma::vec Stdev;
typedef arma::vec Vector;




/***********************************************************
 * rbm parameters
***********************************************************/
typedef struct RestrictedBoltzmannMachinesParamters {
    RestrictedBoltzmannMachinesParamters() :
        N(0), dimension(0),
        n_hidden(10),
        learningRate(0.1),
        regularization(0.0),
        minibatchSize(1),
        CDstep(1), maxStep(100) {};

    unsigned N;
    unsigned dimension;
    unsigned n_hidden;
    double learningRate;
    double regularization;
    unsigned minibatchSize;
    unsigned CDstep;
    unsigned maxStep;
} RBMParameters;




/***********************************************************
 * rbm class (parents class)
***********************************************************/
class RestrictedBoltzmannMachines {
    public:
        void ReadParameters(const string& filename);
        void ParametersSetting(const unsigned& N_, const unsigned& dimension_, const unsigned& n_hidden_,
            const double& learningRate_, const double& regularization_, const unsigned& minibatchSize_, const unsigned& CDstep_, const unsigned& maxStep_);

        void PrintParameters() const;
        void WriteParameters(const string& filename) const;

        unsigned GetN() const;
        unsigned GetDimension() const;

        void PrintWeights() const;
        void PrintBiasVisible() const;
        void PrintBiasHidden() const;
        virtual void PrintResults() const = 0;
        virtual void WriteResults(const string& filename) const = 0;

        virtual void ReadResultFile(const string& filename) = 0;

        virtual void Initialize(const string& initialize_type) = 0;
        void Initialize_Uniform(Weights& weight);
        void Initialize_Gaussian(Weights& weight);

        void NamingFile(string& filename);
        void NamingFileStep(string& filename, const unsigned& step);

        void MiniBathces(arma::field<arma::vec>& minibatch);
        void InitializeDeltaParameters();

       double Sigmoid(const double& x);


    protected:
        Weights weight;                //    weights W_{ij} between visible node (j) and hidden node (i)
        Bias biasVisible;
        Bias biasHidden;

        Weights delta_weight;
        Bias delta_biasVisible;
        Bias delta_biasHidden;

        RBMParameters rbmParas;
};

void RestrictedBoltzmannMachines::ReadParameters(const string& filename) {

    ifstream fin(filename.c_str());
    string s;
    for (unsigned i=0; i<4; i++) getline(fin, s);
    rbmParas.N = stoi(s);

    getline(fin, s);    getline(fin, s);
    rbmParas.dimension = stoi(s);

    getline(fin, s);    getline(fin, s);
    rbmParas.n_hidden = stoi(s);

    getline(fin, s);    getline(fin, s);
    rbmParas.learningRate = stod(s);

    getline(fin, s);    getline(fin, s);
    rbmParas.regularization = stod(s);

    getline(fin, s);    getline(fin, s);
    rbmParas.minibatchSize = stoi(s);

    getline(fin, s);    getline(fin, s);
    rbmParas.CDstep = stoi(s);

    getline(fin, s);    getline(fin, s);
    rbmParas.maxStep = stoi(s);
}


void RestrictedBoltzmannMachines::ParametersSetting(const unsigned& N_, const unsigned& dimension_, const unsigned& n_hidden_,
    const double& learningRate_, const double& regularization_, const unsigned& minibatchSize_, const unsigned& CDstep_, const unsigned& maxStep_) {

    rbmParas.N = N_;
    rbmParas.dimension = dimension_;
    rbmParas.n_hidden = n_hidden_;
    rbmParas.learningRate = learningRate_;
    rbmParas.regularization = regularization_;
    rbmParas.minibatchSize = minibatchSize_;
    rbmParas.CDstep = CDstep_;
    rbmParas.maxStep = maxStep_;
}



void RestrictedBoltzmannMachines::PrintParameters() const {
    cout << "################################################" << endl;
    cout << "##  Restricted Boltzmann Machines Parameters  ##" << endl;
    cout << "################################################" << endl << endl;
    cout << "N: "                                       << rbmParas.N << endl;
    cout << "dimension: "                               << rbmParas.dimension << endl;
    cout << "number of hidden nodes: "                  << rbmParas.n_hidden << endl ;
    cout << "learning rate: "                           << rbmParas.learningRate << endl;
    cout << "regularization rate: "                     << rbmParas.regularization << endl;
    cout << "minibatch size: "                          << rbmParas.minibatchSize << endl;
    cout << "Contrastive Divergence step (k): "         << rbmParas.CDstep << endl;
    cout << "max iteration step: "                      << rbmParas.maxStep << endl << endl;
}

void RestrictedBoltzmannMachines::WriteParameters(const string& filename) const {
    ofstream fsave(filename);
    fsave << "################################################" << endl;
    fsave << "##  Restricted Boltzmann Machines Parameters  ##" << endl;
    fsave << "################################################" << endl << endl;
    fsave << "N: "                                      << rbmParas.N << endl;
    fsave << "dimension: "                              << rbmParas.dimension << endl;
    fsave << "number of hidden nodes: "                 << rbmParas.n_hidden << endl ;
    fsave << "learning rate: "                          << rbmParas.learningRate << endl;
    fsave << "regularization rate: "                    << rbmParas.regularization << endl;
    fsave << "minibatch size: "                         << rbmParas.minibatchSize << endl;
    fsave << "Contrastive Divergence step (k): "        << rbmParas.CDstep << endl;
    fsave << "max iteration step: "                     << rbmParas.maxStep << endl << endl;
}



unsigned RestrictedBoltzmannMachines::GetN() const { return rbmParas.N; }
unsigned RestrictedBoltzmannMachines::GetDimension() const { return rbmParas.dimension; }



void RestrictedBoltzmannMachines::PrintWeights() const {
    cout.precision(8);
    cout.setf(ios::fixed);
    weight.raw_print("weight matrix");
    cout << endl;
}

void RestrictedBoltzmannMachines::PrintBiasVisible() const {
    cout.precision(8);
    cout.setf(ios::fixed);
    biasVisible.raw_print("bias visible");
    cout << endl;
}

void RestrictedBoltzmannMachines::PrintBiasHidden() const {
    cout.precision(8);
    cout.setf(ios::fixed);
    biasHidden.raw_print("bias hidden");
    cout << endl;
}


void RestrictedBoltzmannMachines::Initialize_Uniform(Weights& weight) {

    double minmax = 1.0 / sqrt(weight.n_cols);
//    double minmax = 1.0 / weight.n_cols;
    boost::random::uniform_real_distribution<> uniform_real_dist(-minmax, minmax);      //  Choose a distribution
    boost::random::variate_generator<boost::mt19937 &,
        boost::random::uniform_real_distribution<> > urnd(rng, uniform_real_dist);      //  link the Generator to the distribution

    for (unsigned i=0; i<weight.n_rows; i++)
        for (unsigned j=0; j<weight.n_cols; j++)
            weight(i, j) = urnd();
}


void RestrictedBoltzmannMachines::Initialize_Gaussian(Weights& weight) {

    double std_dev = 1.0 / sqrt(weight.n_cols);
//    double std_dev = 1.0 / weight.n_cols;
    boost::random::normal_distribution<> normal_dist(0.0, std_dev);         //  Choose a distribution
    boost::random::variate_generator<boost::random::mt19937 &,
        boost::random::normal_distribution<> > nrnd(rng, normal_dist);      //  link the Generator to the distribution

    for (unsigned i=0; i<weight.n_rows; i++)
        for (unsigned j=0; j<weight.n_cols; j++)
            weight(i, j) = nrnd();
}



void RestrictedBoltzmannMachines::NamingFile(string& filename) {
    stringstream ss;
    filename += "h";
    ss << rbmParas.n_hidden;
    filename += ss.str();    ss.str("");

    filename += "lr";
    ss << rbmParas.learningRate;
    filename += ss.str();    ss.str("");

    filename += "rg";
    ss << rbmParas.regularization;
    filename += ss.str();    ss.str("");

    filename += ".txt";
}

void RestrictedBoltzmannMachines::NamingFileStep(string& filename, const unsigned& step) {
    stringstream ss;
    filename += "h";
    ss << rbmParas.n_hidden;
    filename += ss.str();    ss.str("");
    
    filename += "lr";
    ss << rbmParas.learningRate;
    filename += ss.str();    ss.str("");

    filename += "rg";
    ss << rbmParas.regularization;
    filename += ss.str();    ss.str("");

    filename += "step";
    ss << step;
    filename += ss.str();    ss.str("");

    filename += ".txt";
}



void RestrictedBoltzmannMachines::MiniBathces(arma::field<arma::vec>& minibatch) {

    boost::random::uniform_real_distribution<> uniform_real_dist(0, 1);        //    Choose a distribution
    boost::random::variate_generator<boost::mt19937 &,
        boost::random::uniform_real_distribution<> > urnd(rng, uniform_real_dist);    //    link the Generator to the distribution

    arma::vec rand_data(rbmParas.N);
    for (unsigned n=0; n<rbmParas.N; n++)
        rand_data(n) = urnd();
    arma::uvec shuffleindex = sort_index(rand_data);

    unsigned n_minibatch = (unsigned) (rbmParas.N / rbmParas.minibatchSize);
    unsigned remainder = rbmParas.N % rbmParas.minibatchSize;
    if ( remainder != 0 ) {
        n_minibatch++;
        minibatch.set_size(n_minibatch);
        unsigned index = 0;
        for (unsigned n=0; n<n_minibatch-1; n++) {
            minibatch(n).set_size(rbmParas.minibatchSize);
            for (unsigned j=0; j<rbmParas.minibatchSize; j++)
                minibatch(n)(j) = shuffleindex(index++);
        }
        minibatch(n_minibatch-1).set_size(remainder);
        for (unsigned j=0; j<remainder; j++)
            minibatch(n_minibatch-1)(j) = shuffleindex(index++);
    }
    else {
        minibatch.set_size(n_minibatch);
        unsigned index = 0;
        for (unsigned n=0; n<n_minibatch; n++) {
            minibatch(n).set_size(rbmParas.minibatchSize);
            for (unsigned j=0; j<rbmParas.minibatchSize; j++)
                minibatch(n)(j) = shuffleindex(index++);
        }
    }
}

void RestrictedBoltzmannMachines::InitializeDeltaParameters() {

    delta_weight.zeros();
    delta_biasVisible.zeros();
    delta_biasHidden.zeros();
}



double RestrictedBoltzmannMachines::Sigmoid(const double& x) {
    return 1.0 / (1.0 + exp(-x));
}










/***********************************************************
 * binary rbm class (child class)
***********************************************************/
class BinaryRBM : public RestrictedBoltzmannMachines {
    public:
        BinaryRBM();
        BinaryRBM(const unsigned& N_, const unsigned& dimension_, const unsigned& n_hidden_,
            const double& learningRate_, const double& regularization_, const unsigned& minibatchSize_, const unsigned& CDstep_, const unsigned& maxStep_);

        void PrintResults() const;
        void WriteResults(const string& filename) const;

        void ReadResultFile(const string& filename);

        void Initialize(const string& initialize_type);

        void Training(df::DataFrame<unsigned>& data);
        void Training(df::DataFrame<unsigned>& data, const unsigned& step);
        void TrainingOneStep(df::DataFrame<unsigned>& data);
        void TrainingMiniBatch(df::DataFrame<unsigned>& data, const Vector& minibatch, double& energy);

        void GibbsSampling(arma::Row<unsigned>& sv, arma::vec& activation_hidden, const arma::Row<unsigned>& v, double& energy);

        void SamplingHiddenNodes_givenV(HiddenNodes& hidden, const arma::Row<unsigned>& sv);
        void SamplingHiddenNodes_givenV(HiddenNodes& hidden, const arma::Row<unsigned>& sv, Vector& activation_hidden, Vector& summation_hidden);
        double ConditionalProb_givenV(const arma::Row<unsigned>& sv, const unsigned& indexi);

        void Reconstruction(arma::Row<unsigned>& sv, const HiddenNodes& hidden);
        double ConditionalProb_givenH(const HiddenNodes& hidden, const unsigned& indexj);

        double Energy(const arma::Row<unsigned>& v, Vector& summation_hidden);
        void WriteEnergy(const string& filename, const double& energy);

        void CumulationDeltaParameters(const arma::Row<unsigned>& vv, const arma::Row<unsigned>& sv, const arma::vec& activation_hidden);
        void UpdateParameter(const unsigned& minibatchSize);
        void Activation_hidden_sample_k(arma::vec& activation_hidden, const arma::Row<unsigned>& sv);
};

BinaryRBM::BinaryRBM() {};
BinaryRBM::BinaryRBM(const unsigned& N_, const unsigned& dimension_, const unsigned& n_hidden_,
    const double& learningRate_, const double& regularization_, const unsigned& minibatchSize_, const unsigned& CDstep_, const unsigned& maxStep_) {

    rbmParas.N = N_;
    rbmParas.dimension = dimension_;
    rbmParas.n_hidden = n_hidden_;
    rbmParas.learningRate = learningRate_;
    rbmParas.regularization = regularization_;
    rbmParas.minibatchSize = minibatchSize_;
    rbmParas.CDstep = CDstep_;
    rbmParas.maxStep = maxStep_;
}


void BinaryRBM::PrintResults() const {
    PrintWeights();
    PrintBiasVisible();
    PrintBiasHidden();
}

void BinaryRBM::WriteResults(const string& filename) const {
    ofstream fsave(filename);
    fsave.precision(8);
    fsave.setf(ios::fixed);
    weight.raw_print(fsave, "weight matrix");
    fsave << endl;
    biasVisible.raw_print(fsave, "bias visible");
    fsave << endl;
    biasHidden.raw_print(fsave, "bias hidden");
    fsave << endl;
    fsave.close();
}

void BinaryRBM::ReadResultFile(const string& filename) {

    weight.set_size(rbmParas.n_hidden, rbmParas.dimension);
    biasVisible.set_size(rbmParas.dimension);
    biasHidden.set_size(rbmParas.n_hidden);

    ifstream fin(filename);
    string dum;
    double value;
    fin >> dum >> dum;
    for (unsigned i=0; i<rbmParas.n_hidden; i++) {
        for (unsigned j=0; j<rbmParas.dimension; j++) {
            fin >> value;
            weight(i, j) = value;
        }
    }
    fin >> dum >> dum;
    for (unsigned j=0; j<rbmParas.dimension; j++) {
        fin >> value;
        biasVisible(j) = value;
    }
    fin >> dum >> dum;
    for (unsigned i=0; i<rbmParas.n_hidden; i++) {
        fin >> value;
        biasHidden(i) = value;
    }
}


void BinaryRBM::Initialize(const string& initialize_type) {

    weight.set_size(rbmParas.n_hidden, rbmParas.dimension);
    biasVisible.set_size(rbmParas.dimension);
    biasHidden.set_size(rbmParas.n_hidden);

    delta_weight.set_size(rbmParas.n_hidden, rbmParas.dimension);
    delta_biasVisible.set_size(rbmParas.dimension);
    delta_biasHidden.set_size(rbmParas.n_hidden);

    if ( initialize_type == "zeros" )
        weight.zeros();
    else if ( initialize_type == "uniform" )
        Initialize_Uniform(weight);
    else if ( initialize_type == "gaussian" )
        Initialize_Gaussian(weight);
    else
        cout << "Usage: you have to type {\"zeros\", \"uniform\", \"gaussian\"}" << endl;

    biasVisible.zeros();
    biasHidden.zeros();
}



void BinaryRBM::Training(df::DataFrame<unsigned>& data) {

    string parafile = "brbm.parameter.";
    NamingFile(parafile);
    WriteParameters(parafile);

    for (unsigned iter=0; iter<rbmParas.maxStep; iter++) {
        cout << "iteration: " << iter << endl;
        double remain_iter = (double) iter / (double) rbmParas.maxStep * 100;
        cout << "remaining iteration ratio: " << remain_iter << "%" << endl << endl;

        TrainingOneStep(data);
    }
    
    string resfile = "brbm.result.";
    NamingFile(resfile);
    WriteResults(resfile);
}

void BinaryRBM::Training(df::DataFrame<unsigned>& data, const unsigned& step) {

    string parafile = "brbm.parameter.";
    NamingFile(parafile);
    WriteParameters(parafile);

    for (unsigned iter=0; iter<rbmParas.maxStep; iter++) {
        cout << "iteration: " << iter << endl;
        double remain_iter = (double) iter / (double) rbmParas.maxStep * 100;
        cout << "remaining iteration ratio: " << remain_iter << "%" << endl << endl;

        TrainingOneStep(data);
    }
    
    string resfile = "brbm.result.";
    NamingFileStep(resfile, step);
    WriteResults(resfile);
}


void BinaryRBM::TrainingOneStep(df::DataFrame<unsigned>& data) {

    arma::field<arma::vec> minibatch;
    MiniBathces(minibatch);

    double energy = 0.0;
    for (unsigned n=0; n<minibatch.size(); n++)
        TrainingMiniBatch(data, minibatch(n), energy);

    energy /= (double) rbmParas.N;

    string energyfile = "brbm.energy.";
    NamingFile(energyfile);
    WriteEnergy(energyfile, energy);
}


void BinaryRBM::TrainingMiniBatch(df::DataFrame<unsigned>& data, const Vector& minibatch, double& energy) {

    InitializeDeltaParameters();

    for (unsigned n=0; n<minibatch.size(); n++) {

        //  Pick one record
        arma::Row<unsigned> rawVisible = data.GetDataRow(minibatch(n));
        arma::Row<unsigned> sampleVisible = rawVisible;

        arma::vec activation_hidden(rbmParas.n_hidden);

        //    GibbsSampling
        GibbsSampling(sampleVisible, activation_hidden, rawVisible, energy);

        //    Cumulation of delta_weight and delta_bias in minibatch
        CumulationDeltaParameters(rawVisible, sampleVisible, activation_hidden);
    }

    //    Update Parameters
    UpdateParameter(minibatch.size());
}




void BinaryRBM::GibbsSampling(arma::Row<unsigned>& sv, arma::vec& activation_hidden, const arma::Row<unsigned>& v, double& energy) {

    HiddenNodes hidden(rbmParas.n_hidden);
    Vector summation_hidden(rbmParas.n_hidden);

    SamplingHiddenNodes_givenV(hidden, sv, activation_hidden, summation_hidden);
    Reconstruction(sv, hidden);

    for (unsigned gibbssampling=1; gibbssampling<rbmParas.CDstep; gibbssampling++) {
        SamplingHiddenNodes_givenV(hidden, sv);
        Reconstruction(sv, hidden);
    }
    energy += Energy(v, summation_hidden);
}



void BinaryRBM::SamplingHiddenNodes_givenV(HiddenNodes& hidden, const arma::Row<unsigned>& sv) {
    boost::random::uniform_real_distribution<> uniform_real_dist(0, 1);        //    Choose a distribution
    boost::random::variate_generator<boost::mt19937 &,
        boost::random::uniform_real_distribution<> > urnd(rng, uniform_real_dist);    //    link the Generator to the distribution

    for (unsigned i=0; i<rbmParas.n_hidden; i++) {
        hidden(i) = ConditionalProb_givenV(sv, i) > urnd() ? 1 : 0;
    }
}

void BinaryRBM::SamplingHiddenNodes_givenV(HiddenNodes& hidden, const arma::Row<unsigned>& sv, Vector& activation_hidden, Vector& summation_hidden) {
    boost::random::uniform_real_distribution<> uniform_real_dist(0, 1);        //    Choose a distribution
    boost::random::variate_generator<boost::mt19937 &,
        boost::random::uniform_real_distribution<> > urnd(rng, uniform_real_dist);    //    link the Generator to the distribution

    for (unsigned i=0; i<rbmParas.n_hidden; i++) {
        summation_hidden(i) = ConditionalProb_givenV(sv, i);
        activation_hidden(i) = Sigmoid(summation_hidden(i));
        hidden(i) = activation_hidden(i) > urnd() ? 1 : 0;
    }
}

double BinaryRBM::ConditionalProb_givenV(const arma::Row<unsigned>& sv, const unsigned& indexi) {
    return arma::dot(weight.row(indexi), sv) + biasHidden(indexi);
}


void BinaryRBM::Reconstruction(arma::Row<unsigned>& sv, const HiddenNodes& hidden) {
    boost::random::uniform_real_distribution<> uniform_real_dist(0, 1);        //    Choose a distribution
    boost::random::variate_generator<boost::mt19937 &,
        boost::random::uniform_real_distribution<> > urnd(rng, uniform_real_dist);    //    link the Generator to the distribution

    for (unsigned j=0; j<rbmParas.dimension; j++) {
        sv(j) = ConditionalProb_givenH(hidden, j) > urnd() ? 1 : 0;
    }
}

double BinaryRBM::ConditionalProb_givenH(const HiddenNodes& hidden, const unsigned& indexj) {
    return Sigmoid(arma::dot(weight.col(indexj), hidden) + biasVisible(indexj));
}




double BinaryRBM::Energy(const arma::Row<unsigned>& v, Vector& summation_hidden) {
    return - arma::dot(v, biasVisible) - arma::sum(log(1.0 + exp(summation_hidden)));
}

void BinaryRBM::WriteEnergy(const string& filename, const double& energy) {
    ofstream fsave(filename, fstream::out | fstream::app);
    fsave << energy << endl;
    fsave.close();
}



void BinaryRBM::CumulationDeltaParameters(const arma::Row<unsigned>& rv, const arma::Row<unsigned>& sv, const arma::vec& activation_hidden) {

    arma::vec activation_hidden_sample_k(rbmParas.n_hidden);
    Activation_hidden_sample_k(activation_hidden_sample_k, sv);

//    delta_weight += activation_hidden * rv - activation_hidden_sample_k * sv;
//    for loops are faster than matrix multiplication
	for (unsigned i=0; i<rbmParas.n_hidden; i++)
		for (unsigned j=0; j<rbmParas.dimension; j++)
			delta_weight(i, j) += activation_hidden(i) * (double) rv(j) - activation_hidden_sample_k(i) * (double) sv(j);

    for (unsigned j=0; j<rbmParas.dimension; j++)
        delta_biasVisible(j) += (double) rv(j) - (double) sv(j);

    delta_biasHidden += activation_hidden - activation_hidden_sample_k;
}

void BinaryRBM::UpdateParameter(const unsigned& minibatchSize) {

    weight *= (1.0 - rbmParas.regularization * rbmParas.learningRate / (double) rbmParas.N);
    weight += rbmParas.learningRate * delta_weight / (double) minibatchSize;
    biasVisible += rbmParas.learningRate * delta_biasVisible / (double) minibatchSize;
    biasHidden += rbmParas.learningRate * delta_biasHidden / (double) minibatchSize;
}



void BinaryRBM::Activation_hidden_sample_k(arma::vec& activation_hidden, const arma::Row<unsigned>& sv) {
    for (unsigned i=0; i<rbmParas.n_hidden; i++)
        activation_hidden(i) = Sigmoid(arma::dot(weight.row(i), sv) + biasHidden(i));
}










/***********************************************************
 * gaussian bernoulli rbm class (child class)
***********************************************************/
class GaussianBernoulliRBM : public RestrictedBoltzmannMachines {
    public:
        GaussianBernoulliRBM();
        GaussianBernoulliRBM(const unsigned& N_, const unsigned& dimension_, const unsigned& n_hidden_,
            const double& learningRate_, const double& regularization_, const unsigned& minibatchSize_, const unsigned& CDstep_, const unsigned& maxStep_);

        void PrintStdev() const;
        void PrintResults() const;
        void WriteResults(const string& filename) const;

        void ReadResultFile(const string& filename);


        void Initialize(const string& initialize_type);

        void Training(df::DataFrame<double>& data);
        void Training(df::DataFrame<double>& data, const unsigned& step);
        void TrainingOneStep(df::DataFrame<double>& data);
        void TrainingMiniBatch(df::DataFrame<double>& data, const Vector& minibatch, double& energy);

        void GibbsSampling(arma::Row<double>& sv, arma::vec& activation_hidden, const arma::Row<double>& v, double& energy);

        void SamplingHiddenNodes_givenV(HiddenNodes& hidden, const arma::Row<double>& sv);
        void SamplingHiddenNodes_givenV(HiddenNodes& hidden, const arma::Row<double>& sv, Vector& activation_hidden, Vector& summation_hidden);
        double ConditionalProb_givenV(const arma::Row<double>& sv, const unsigned& indexi);

        void Reconstruction(arma::Row<double>& sv, HiddenNodes& hidden);
        double ConditionalProb_givenH(const HiddenNodes& hidden, const unsigned& indexi);

        double Energy(const arma::Row<double>& v, Vector& summation_hidden);
        void WriteEnergy(const string& filename, const double& energy);

        void CumulationDeltaParameters(const arma::Row<double>& rv, const arma::Row<double>& sv, const arma::vec& activation_hidden);
        void UpdateParameter(const unsigned& minibatchSize);
        void Activation_hidden_sample_k(arma::vec& activation_hidden, const arma::Row<double>& sv);



    private:
        Stdev stdev;
        Stdev delta_stdev;
};

GaussianBernoulliRBM::GaussianBernoulliRBM() {};
GaussianBernoulliRBM::GaussianBernoulliRBM(const unsigned& N_, const unsigned& dimension_, const unsigned& n_hidden_, 
    const double& learningRate_, const double& regularization_, const unsigned& minibatchSize_, const unsigned& CDstep_, const unsigned& maxStep_) {

    rbmParas.N = N_;
    rbmParas.dimension = dimension_;
    rbmParas.n_hidden = n_hidden_;
    rbmParas.learningRate = learningRate_;
    rbmParas.regularization = regularization_;
    rbmParas.minibatchSize = minibatchSize_;
    rbmParas.CDstep = CDstep_;
    rbmParas.maxStep = maxStep_;
}


void GaussianBernoulliRBM::PrintStdev() const {
    cout.precision(8);
    cout.setf(ios::fixed);
    stdev.raw_print("standard deviation");
    cout << endl;
}

void GaussianBernoulliRBM::PrintResults() const {
    PrintWeights();
    PrintBiasVisible();
    PrintBiasHidden();
    PrintStdev();
}

void GaussianBernoulliRBM::WriteResults(const string& filename) const {
    ofstream fsave(filename);
    fsave.precision(8);
    fsave.setf(ios::fixed);
    weight.raw_print(fsave, "weight matrix");
    fsave << endl;
    biasVisible.raw_print(fsave, "bias visible");
    fsave << endl;
    biasHidden.raw_print(fsave, "bias hidden");
    fsave << endl;
    stdev.raw_print(fsave, "standard deviation");
    fsave << endl;
    fsave.close();
}



void GaussianBernoulliRBM::ReadResultFile(const string& filename) {

    weight.set_size(rbmParas.dimension, rbmParas.n_hidden);
    biasVisible.set_size(rbmParas.dimension);
    biasHidden.set_size(rbmParas.n_hidden);
    stdev.set_size(rbmParas.dimension);

    ifstream fin(filename);
    string dum;
    double value;
    fin >> dum >> dum;
    for (unsigned i=0; i<rbmParas.n_hidden; i++) {
        for (unsigned j=0; j<rbmParas.dimension; j++) {
            fin >> value;
            weight(i, j) = value;
        }
    }
    fin >> dum >> dum;
    for (unsigned j=0; j<rbmParas.dimension; j++) {
        fin >> value;
        biasVisible(j) = value;
    }
    fin >> dum >> dum;
    for (unsigned i=0; i<rbmParas.n_hidden; i++) {
        fin >> value;
        biasHidden(i) = value;
    }
    fin >> dum >> dum;
    for (unsigned j=0; j<rbmParas.dimension; j++) {
        fin >> value;
        stdev(j) = value;
    }
}



void GaussianBernoulliRBM::Initialize(const string& initialize_type) {

    weight.set_size(rbmParas.n_hidden, rbmParas.dimension);
    biasVisible.set_size(rbmParas.dimension);
    biasHidden.set_size(rbmParas.n_hidden);
    stdev.set_size(rbmParas.dimension);

    delta_weight.set_size(rbmParas.n_hidden, rbmParas.dimension);
    delta_biasVisible.set_size(rbmParas.dimension);
    delta_biasHidden.set_size(rbmParas.n_hidden);
    delta_stdev.set_size(rbmParas.dimension);

    if ( initialize_type == "zeros" )
        weight.zeros();
    else if ( initialize_type == "uniform" )
        Initialize_Uniform(weight);
    else if ( initialize_type == "gaussian" )
        Initialize_Gaussian(weight);
    else
        cout << "Usage: you have to type {\"zeros\", \"uniform\", \"gaussian\"}" << endl;

    biasVisible.zeros();
    biasHidden.zeros();
    stdev.ones();
}




void GaussianBernoulliRBM::Training(df::DataFrame<double>& data) {

    string parafile = "gbrbm.parameter.";
    NamingFile(parafile);
    WriteParameters(parafile);

    for (unsigned iter=0; iter<rbmParas.maxStep; iter++) {
        cout << "iteration: " << iter << endl;
        double remain_iter = (double) iter / (double) rbmParas.maxStep * 100;
        cout << "remaining iteration ratio: " << remain_iter << "%" << endl << endl;

        TrainingOneStep(data);
    }
    
    string resfile = "gbrbm.result.";
    NamingFile(resfile);
    WriteResults(resfile);
}

void GaussianBernoulliRBM::Training(df::DataFrame<double>& data, const unsigned& step) {

    string parafile = "gbrbm.parameter.";
    NamingFile(parafile);
    WriteParameters(parafile);

    for (unsigned iter=0; iter<rbmParas.maxStep; iter++) {
        cout << "iteration: " << iter << endl;
        double remain_iter = (double) iter / (double) rbmParas.maxStep * 100;
        cout << "remaining iteration ratio: " << remain_iter << "%" << endl << endl;

        TrainingOneStep(data);
    }
    
    string resfile = "gbrbm.result.";
    NamingFileStep(resfile, step);
    WriteResults(resfile);
}


void GaussianBernoulliRBM::TrainingOneStep(df::DataFrame<double>& data) {

    arma::field<arma::vec> minibatch;
    MiniBathces(minibatch);

    double energy = 0.0;
    for (unsigned n=0; n<minibatch.size(); n++)
        TrainingMiniBatch(data, minibatch(n), energy);

    energy /= (double) rbmParas.N;

    string energyfile = "gbrbm.energy.";
    NamingFile(energyfile);
    WriteEnergy(energyfile, energy);
}


void GaussianBernoulliRBM::TrainingMiniBatch(df::DataFrame<double>& data, const Vector& minibatch, double& energy) {

    InitializeDeltaParameters();

    for (unsigned n=0; n<minibatch.size(); n++) {

        //  Pick one record
        arma::Row<double> rawVisible = data.GetDataRow(minibatch(n));
        arma::Row<double> sampleVisible = rawVisible;

        arma::vec activation_hidden(rbmParas.n_hidden);

        //    GibbsSampling
        GibbsSampling(sampleVisible, activation_hidden, rawVisible, energy);

        //    Cumulation of delta_weight and delta_bias in minibatch
        CumulationDeltaParameters(rawVisible, sampleVisible, activation_hidden);
    }

    //    Update Parameters
    UpdateParameter(minibatch.size());
}





void GaussianBernoulliRBM::GibbsSampling(arma::Row<double>& sv, arma::vec& activation_hidden, const arma::Row<double>& v, double& energy) {

    HiddenNodes hidden(rbmParas.n_hidden);
    Vector summation_hidden(rbmParas.n_hidden);

    SamplingHiddenNodes_givenV(hidden, sv, activation_hidden, summation_hidden);
    Reconstruction(sv, hidden);

    for (unsigned gibbssampling=1; gibbssampling<rbmParas.CDstep; gibbssampling++) {
        SamplingHiddenNodes_givenV(hidden, sv);
        Reconstruction(sv, hidden);
    }
    energy += Energy(v, summation_hidden);
}


void GaussianBernoulliRBM::SamplingHiddenNodes_givenV(HiddenNodes& hidden, const arma::Row<double>& sv) {
    boost::random::uniform_real_distribution<> uniform_real_dist(0, 1);        //    Choose a distribution
    boost::random::variate_generator<boost::mt19937 &,
        boost::random::uniform_real_distribution<> > urnd(rng, uniform_real_dist);    //    link the Generator to the distribution

    for (unsigned i=0; i<rbmParas.n_hidden; i++) {
        hidden(i) = ConditionalProb_givenV(sv, i) > urnd() ? 1 : 0;
    }
}

void GaussianBernoulliRBM::SamplingHiddenNodes_givenV(HiddenNodes& hidden, const arma::Row<double>& sv, Vector& activation_hidden, Vector& summation_hidden) {
    boost::random::uniform_real_distribution<> uniform_real_dist(0, 1);        //    Choose a distribution
    boost::random::variate_generator<boost::mt19937 &,
        boost::random::uniform_real_distribution<> > urnd(rng, uniform_real_dist);    //    link the Generator to the distribution

    for (unsigned i=0; i<rbmParas.n_hidden; i++) {
        summation_hidden(i) = ConditionalProb_givenV(sv, i);
        activation_hidden(i) = Sigmoid(summation_hidden(i));
        hidden(i) = activation_hidden(i) > urnd() ? 1 : 0;
    }
}

double GaussianBernoulliRBM::ConditionalProb_givenV(const arma::Row<double>& sv, const unsigned& indexi) {
//    assume standard deviations are 1
//    return arma::dot(weight.row(indexi), sv / stdev.t()) + biasHidden(indexi);
    return arma::dot(weight.row(indexi), sv) + biasHidden(indexi);
}



void GaussianBernoulliRBM::Reconstruction(arma::Row<double>& sv, HiddenNodes& hidden) {
    for (unsigned j=0; j<rbmParas.dimension; j++)
        sv(j) = ConditionalProb_givenH(hidden, j);
}

double GaussianBernoulliRBM::ConditionalProb_givenH(const HiddenNodes& hidden, const unsigned& indexj) {
//    assume standard deviations are 1
//    double mean = arma::dot(weight.col(indexj), hidden) * stdev(indexj) + biasVisible(indexj);
    double mean = arma::dot(weight.col(indexj), hidden) + biasVisible(indexj);

    boost::random::normal_distribution<> normal_dist(mean, stdev(indexj));      //    Choose a distribution
    boost::random::variate_generator<boost::random::mt19937 &,
        boost::random::normal_distribution<> > nrnd(rng, normal_dist);          //    link the Generator to the distribution

    return nrnd();
}


double GaussianBernoulliRBM::Energy(const arma::Row<double>& v, Vector& summation_hidden) {
//    assume standard deviations are 1
    return - arma::dot(v, biasVisible) - arma::sum(log(1.0 + exp(summation_hidden)));
}

void GaussianBernoulliRBM::WriteEnergy(const string& filename, const double& energy) {
    ofstream fsave(filename, fstream::out | fstream::app);
    fsave << energy << endl;
    fsave.close();
}




void GaussianBernoulliRBM::CumulationDeltaParameters(const arma::Row<double>& rv, const arma::Row<double>& sv, const arma::vec& activation_hidden) {

    arma::vec activation_hidden_sample_k(rbmParas.n_hidden);
    Activation_hidden_sample_k(activation_hidden_sample_k, sv);
    
//    assume standard deviations are 1
//    delta_weight += activation_hidden * (rv / stdev.t()) - activation_hidden_sample_k * (sv / stdev.t());
//    delta_weight += activation_hidden * rv - activation_hidden_sample_k * sv;
//    for loops are faster than matrix multiplication
	for (unsigned i=0; i<rbmParas.n_hidden; i++)
		for (unsigned j=0; j<rbmParas.dimension; j++)
//			delta_weight(i, j) += activation_hidden(i) * (double) rv(j) / stdev(j) - activation_hidden_sample_k(i) * (double) sv(j) / stdev(j);
			delta_weight(i, j) += activation_hidden(i) * (double) rv(j) - activation_hidden_sample_k(i) * (double) sv(j);

//    biasVisible += (rv - sv).t() / stdev / stdev;
    delta_biasVisible += (rv - sv).t();

    delta_biasHidden += activation_hidden - activation_hidden_sample_k;
}



void GaussianBernoulliRBM::UpdateParameter(const unsigned& minibatchSize) {

    weight *= (1.0 - rbmParas.regularization * rbmParas.learningRate / (double) rbmParas.N);
    weight += rbmParas.learningRate * delta_weight / (double) minibatchSize;
    biasVisible += rbmParas.learningRate * delta_biasVisible / (double) minibatchSize;
    biasHidden += rbmParas.learningRate * delta_biasHidden / (double) minibatchSize;
}


void GaussianBernoulliRBM::Activation_hidden_sample_k(arma::vec& activation_hidden, const arma::Row<double>& sv) {
//    assume standard deviations are 1
    for (unsigned i=0; i<rbmParas.n_hidden; i++)
//        activation_hidden(i) = Sigmoid(arma::dot(weight.row(i), sv / stdev.t()) + biasHidden(i));
        activation_hidden(i) = Sigmoid(arma::dot(weight.row(i), sv) + biasHidden(i));
}




}

#endif
