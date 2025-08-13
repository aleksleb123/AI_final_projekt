# Alzheimers Diagnosis Neural Net Project Rewrite

This code is the current version for the Alzheimers CNN uncertanity estimation project. The project consists of a set of main scripts (in the root folder) and some utilities. In order to use the project:

1. Edit "config.toml" with the details of the ensemble that you would like to train (size, name, epochs etc). Make sure that the model name and ensemble name are the same if you'd like to run the ensemble analysis later.
2. Run "train_cnn.py". This will train and save a new ensemble of CNN models using the name and congfiguration options given in the config file.
3. Run "ensemble_predict.py". This will generate the predictions of the models on the test and validation datasets and save them to the model ensemble folder. 
4. Run "threshold_xarray.py". This run some analysis on the ensemble and generates a set of graphs and statistics. 

"bayesian.py" is unfinished and does not currently work. The other two threshold files are old implementations. 'sensitivity_analysis.py' can be optionally used to generate some model number sensitivity data, but the implementation is buggy currently. Apologies for the messy code throughout! 