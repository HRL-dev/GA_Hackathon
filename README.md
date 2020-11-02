# General Assembly Hackathon

## Problem Statement

The question I'll be trying to answer with this dataset is as follows: can I accurately determine whether a pet in an animal shelter will be adopted, transfered to a partner shelter, returned to their owner, euthanized, or die in the shelter of other causes?

## Process Description

I began with general EDA and cleaning, determined the baseline model would be 40% accuracy if we predicted "Adoption" for every outcome.

My first efforts were to determine the type of the data and whether there were null values. The type of every column was object and there were null values, so I imputed and/or dropped the nulls (based on frequency) and moved on to determining whether I could turn the object variables into float/int. First, I turned the DateTime column into pd.DateTime, then feature engineered three columns - year_month, year, and Month - that I would be able to use more easily than the exact, individual date and time.

Next, I filtered by cardinality, in order to rule out columns that were simply too varied to dummify. However, I was determined to retain "Breed," which I felt would be an important determining factor, so I ended up with quite a few features nonethless.

Because of the large number of features and my dummified y-variable matrix, the first model I used was a neural network with an input layer and two hidden layers (all using relu activation function), and an output layer using softmax activation function. Because I mostly care about whether I am able to accurately categorize the outcomes of these pets, I input the accuracy metric when I compiled the model. I used the categorical cross-entropy loss function.

My model gave a validation score of 61% - significantly better than the baseline accuracy - but still not great. Additionally, my validation loss was much higher than my training loss, suggesting that my model was overfitting. I decided to try the Dropout method.

I instantiated almost the exact same model, except with only one hidden layer and a dropout layer with .2, but the validation score was still poor. As I increased the dropout, the validation score did not improve upon the prior model, so I moved on to another model.

I decided to use Random Forest. In order to use Random Forest, I had to go back to the original, un-dummified y variable and map a function onto it that turned the five options into 1-5, because Random Forest only accepts 1D arrays.

After instantiating Random Forest and gridsearching over the best parameters (n_estimators and max_features), 
I was disappointed to find that the best score was worse than the prior scores.

I decided to take a different approach: instead of working with hundreds of features thanks to my dummified 'Breed' column, I was going to try to scale that down a bit. I found [this kaggle notebook](https://www.kaggle.com/uchayder/take-a-look-at-the-data) extremely helpful. I turned the 'Breed' column into mix vs. purebred, and made the Age column all in floats relating to year-length, rather than '3 weeks' or '5 days.' 

With this much smaller dataset, I ran a neural network again, only to get a validation score of 62%. In the end, after several models, I determined that while this is not perfect, it is significantly better than the baseline accuracy score, and my training and validation loss are much more similar now, meaning that I'm no longer overfitting.


## Conclusions & Recommendations

My conclusion, based on the models that I ran, is that age is the most important determining factor in whether or not an animal is adopted from a shelter, but that it's often just luck for the animals.

For anyone who wants to improve upon this model, it's almost definitely possible - it may just require more time than I had today, or tools that I either don't have yet or didn't think of. 