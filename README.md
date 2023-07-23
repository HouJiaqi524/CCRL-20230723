# CCRL

## Name
Copula based common representation learning (CCRL).

## Description
Code of the architecture of CCRL as well as some experiments, implemented by python 3.7 and pytorch 1.8(or 1.10) .
* CCRL.py: architecture of CCRL for MNIST transfer learning and XRMB data fution classification.
* MonotonicNN.py, NeuralIntegral.py, ParallelNeuralIntegral.py are from https://github.com/AWehenkel/UMNN/tree/master/models/UMNN, with little modification, to bulid MonotonicNNs in CCRL. 
* Dataloader.py: load MNIST and two-view XRMB datasets, first to down load the datasets provided in the links in the file, and place them in the same directory as Dataloader.py .
* train_model.py: model training for MNIST and XRMB
* objective.py: CCRL's loss function
* main4test.py: train CCRL representations of two views.
* successive_classification_XRMB.py: load pretrained TD-CCRL for original feature transformation, then lda classification is conducted by using concatenated transformed features.
* successive_classification_MNIST.py: load pretrained CCRL(TD-CCRL) for original feature transformation, then SVM classification is conducted by using concatenated transformed features.
* utils_.py: some basic functions
* simulated_experiment1: experiments of CCRL, DCA and DCCA on a testbed
* simulated_experiment2: CCRL's classification case on concentric circles and two-moons dataset.
* cancer_surviaval_analysis: CCRL to extract multi_omics features related to survival info on LAML cancer dataset from TCGA, we used the preprocessed matrix data from DeepProg.
* out: saved pretrained CCRL(TD-CCRL) model and transformed features

 
## Usage
### 1. retrained CCRL(TD-CCRL) model
* main4test.py: change the hyper params in loss function of CCRL if needed
* train_model.py: find the TODO part to choose to conduct model training on XRMB or MNIST, and the corresponding task(None, 'L2R', 'R2L', 'SupervisedDataFusion', tasks will show their difference in CCRL.py).

### 2. load the pretrained CCRL(TD-CCRL) model for successive classification
* successive_classification_MNIST.py, successive_classification_XRMB.py: find the TODO part, to load the pretrained models, to conduct classification.


## Support
/

## Authors and acknowledgment
/

## Project status
/
