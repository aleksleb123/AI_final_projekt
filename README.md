# Alzheimers Diagnosis Neural Net Project Rewrite

1. Edit "config.toml" with the details of the ensemble that you would like to train (size, name, epochs etc). Make sure that the model name and ensemble name are the same if you'd like to run the ensemble analysis later.
2. Run "train_cnn.py". This will train and save a new ensemble of CNN models using the name and congfiguration options given in the config file.
3. Run "ensemble_predict.py". This will generate the predictions of the models on the test and validation datasets and save them to the model ensemble folder. 
4. Run "threshold_xarray.py". This run some analysis on the ensemble and generates a set of graphs and statistics. 

Navodila uporabe:
  1. Downlodaj vse
  2. Jaz uporabljam PyCharm, kot vmesnik za python in imam vse v enem direktoriju izgleda kot je na sliki.<img width="666" height="551" alt="Izgled direktorija" src="https://github.com/user-attachments/assets/29d42d1c-3beb-4113-8b60-4ad5eaaa4fce" />
  
