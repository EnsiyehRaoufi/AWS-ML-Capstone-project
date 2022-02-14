
# Capstone Starbucks project

I chose Starbucks capstone challenge, because the information gathered in this dataset was so interesting to me. 

## Project Set Up and Installation
We should install the required packages. In this project I've upgraded the NumPy package to ensure that it can read JSON files correctly and, I've installed Autoguon that can use its tabular predictor at the end of the program. I ran my program using an AWS ml.t3.medium notebook instance. We should install and import necessary packages. In my case I've imported these libraries:

import datetime
import time
import tarfile
import math
import json
import boto3
import random
import pandas as pd
import numpy as np
from sagemaker import get_execution_role
import sagemaker
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

## Dataset

### Overview
I mention the information that Udacity gave me first. Then I will explain my method for creating an input for my model training. Here is the Udacity description on this dataset:

#### Udacity dataset description

The data is contained in three files:

* portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
* profile.json - demographic data for each customer
* transcript.json - records for transactions, offers received, offers viewed, and offers completed

Here is the schema and explanation of each variable in the files:

**portfolio.json**

* id (string) - offer id
* offer_type (string) - type of offer ie BOGO, discount, informational
* difficulty (int) - minimum required spend to complete an offer
* reward (int) - reward given for completing an offer
* duration (int) - time for offer to be open, in days
* channels (list of strings)

**profile.json**

* age (int) - age of the customer 
* became_member_on (int) - date when customer created an app account
* gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
* id (str) - customer id
* income (float) - customer's income

**transcript.json**

* event (str) - record description (ie transaction, offer received, offer viewed, etc.)
* person (str) - customer id
* time (int) - time in hours since start of test. The data begins at time t=0
* value - (dict of strings) - either an offer id or transaction amount depending on the record

#### My changes on data and creating a suitable input 

I read all the json files.

By watching the transcript table carefully we can see that the "event" column consists of a categorical data with 'offer completed', 'offer received', 'offer viewed' and  'transaction' as its category names. So if we care about this feature to be one of our features for training the model or other future works on the dataset, we should generate a label encoding scheme for mapping each category to a numeric value. But because our category is not ordinal and for example if we assign the number 1 to the 'offer completed' then it has no preference to the number 4 that we assigned to the 'transaction' category, so we should do the one-hot encoding scheme that encodes or transforms this feature into 4 binary features which can only contain a value of 1 or 0. Also the "value" column contains some dictionary type data that its keys should be extracted and turn into some other new columns.

For the portfolio table we should also do the one-hot encoding on the "channels" and "offer_type" columns. We should also notice that the data on the "channels" column is of list type.

In the profile table we should be careful about NaN cells and choose a strategy for filling or removing the rows that contain NaN. I prefer to fill the NaN cells of a column with the help of the other cells of the column. For the gender column I prefer to fill the NaN cells with a random choose of "F" or "M". We can ignore the Other genders that mentiond by "O" because of their minority in the dataset. I also prefer to fill the NaN cells of the "income" column with the average of the column. We should change the cells that are filled with 118 on the "age" column because they are representing the NaN cells. For this column I also choose the average amount of the cells that are not 118, to fill these cells.

### Access
I've downloaded all the json files of Starbucks dataset from Udacity. I have uploaded all the JSON files of the dataset as "data.zip" to this direstory.

## Model Training
My problem is one of the hundreds of question that one can ask about this dataset and here is the question I am asking:

**"Build a machine learning model that predicts how much more someone will spend based on demographics and offer type they will receive/view during the time.". **
 
For solving this problem I should first make a complete table from all other data tables that can work on it as the source data. This table must contain all the persons with all of their offers and transactions infromation. Then I should make a column that contains cumulative amount of transacrtions in this table. Because I'm investigating the increasing amount of customer transactions when they interface an offer, so I should remove the other rows that a customer does not receive/view/complete an offer. Then I should use a model like Random Forest Scikit_Learn to predict the customer's cumulative transactions. I can ignore columns that have a less importance by several training and getting feature importance each time. I should evaluate my model using metrics like R2 score, mean-squared/absolute error that are suitable metrics for my regression problem.

## Machine Learning Pipeline
The solution to my question would be found with building a machine learning model that predicts the cumulative transactions of persons who received at least one offer at the times of they receive, view or complete an offer (just 5 persons of 17000 does not received any offers). This is a regression problem and it's enough to create a model that predicts my target (cumulative transactions) with the minimum errors. As an early plan, I can use random forest model of the Scikit_Learn library to train my data. I can use Amazon Sagemaker Estimator to do a training job. I should fine tune the estimator to find the best hyperparameters before the final training. We know that the hyperparameters in this model except for the "features" and "target", only include "the number of random records from the dataset" and "the number of trees in the algorithm". So after fine tuning and finding the best model, we should do a training job using a Sagemaker Sklearn Estimator. Then we should deploy our model to an endpoint and get the predictions on the test dataset by invoking the endpoint. We should evaluate our model using some appropriate metrics for a regression problem. We can use the R2 score that shows the performance of our model well. We should delete the endpoint after use to avoid incuring costs. 

## Standout Suggestions
I think it's a good idea to develop a deep neural network model to predict my test data with an R2 score of above 90%. Also by having a neat table of data, I consider designing various classification and regression problems on the same dataset that can answer other aspects of the effects of advertisement on sales. For example, we can create a binary classification model to predict if a customer receives a special kind of offer, would they complete the offer or not, or will they view the offer or not (predicting a class label). In the case of predicting a class probability, we can ask what are the probabilities of a specified customer to complete any of 10 classes of offers. There are many problems that we can design and answer on this dataset. Solving each of them needs a lot of effort, but at last your hard work always pays off.

##License
I have attached a Apache License 2.0 to this repository, click to see [here](https://github.com/EnsiyehRaoufi/AWS-ML-Capstone-project/blob/main/LICENSE)

