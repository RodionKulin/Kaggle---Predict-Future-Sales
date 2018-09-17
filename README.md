# Kaggle - Predict-Future-Sales
Solving Kaggle machine learning task to Predict Future Sales.


# Task
https://www.kaggle.com/c/competitive-data-science-predict-future-sales

Provided dataset is challenging time-series sales data on a daily interval.
Task is to predict total sales for every product and store in the next month. 


# Project description
## Prepare data

-Convert data to supervised learning data. 

-Split data on training and validation

-Create day of year and year features for datetime.
Normalize them.

-Convert store ids to OHE format.

-Convert item ids to OHE format.

-Normalize sales values

-Merge features into flat array. So it can be used as input to LSTM model.

-Split data into small sequences that will be fed to LSTM model.
Use Sequence that acts similar to python generator.


## Model
-Pick model: LSTM.
Model predicts sales for one next day at a time. 
Repeat to predict sales for next target number of days.

-Load existing model weights. Used if continue training existing model.

-Configure early stopping. So training finishes when no significant improvement is achived any more.

-Configure checkpoints. So can continue training after unexpected interruption.

-Train on training data and evaluate prediction loss on validation data.


## Save results
-Save results to files:
Data encoders and scalers;
Best model during training;
Evaluation metrics;
Experiment description: train time, data size, model summary, sanity check results;
Output log.

-Store resulting files in Amazon S3 


## Run 
-Use experiment runner to:

1)Chain running multiple experiments with different data preperation techniques and models in AWS EC2.

2)Shut down EC2 instance after all experiments are finished.


# Configuration parameters
-Directory in S3
aws_experiment_dir

-Description of model
experiment_description

-Training speed
batch_size = 8
epochs = 25
steps_per_epoch = 1000
steps_per_epoch_val = 200

-Model
input_length = 50
