* trail_design.py is the main code. Here, we use an orthogonal design for automatic parameter tuning experiments
* obtain_trail_index.py and  class "ObtainTrialIndex()" to conduct one process under specified hyperparameters, including training CCRL for multi-omics feature extraction, clustering obtained abstract multi-omics features to several survival types, obtaining the performance evaluation index like p-value and C-index.
* pdf_KM_plot.py to plot KM curves.
* preprocess_and_statistics.py is a preprocess module to normalize data and out some statistics of the data. 