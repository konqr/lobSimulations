This folder is an update for the original versions in directory 'lobAE'.

It mainly contains the following changes:
1) Percentile has been added to positions.
2) The encoder now also takes in S(n-1) and U(n-1).
3) A utils.py file is created to store all the common functions used for different architechtures of AE.
Files with the above updates, i.e. the new version of 'datasetCreation.py' and 'AE_simple.py', are named as 'datasetCreation_v1.py' and 'AE_simple_v1.py'

What to do next:
1) Tune hyperparameters like state_dim, num_layers, batch_size and learning_rate, see if there will be better performance.
2) Better loss function choices.
3) Include more history and feed more data to the model.
