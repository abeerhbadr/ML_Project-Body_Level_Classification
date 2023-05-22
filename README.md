# Body-Level-classification

# **Machine Learning Project**

## Contributors Names
| **Name** |
| --- |
| Abeer Hussein |
| Ghiath Ajam |
| Mohamed Akram |
|Nour-Aldin Mustafa |

## **I. Data Analysis**
**Columns excluding label:** (blue for **Categorical** , red for **Numerical** )
| Gender | H\_Cal\_Consump | Alchohol\_Consump | Food\_Between\_Meals |
| --- | --- | --- | --- |
| Fam\_Hist | H\_Cal\_Burn | Transport | Smoking |
| Age | Height | Weight | Veg\_Consump |
| Water\_Consump | Meal\_Count | Phys\_Act | Time\_E\_Dev |

Categorical features are either:

- boolean (yes/no OR male/female)
- frequency (no/sometimes/frequently/always)

Except for Transport column which contains the type of transportation a person uses

['Public\_Transportation' 'Automobile' 'Walking' 'Bike' 'Motorbike']

We encode these columns with proper numbers to better visualize and analyze relations.

No / Sometimes / Frequently / Always =\> 0 / 1 / 2 / 3

Bike / Walking / Public\_Transport / MotorBike / Automobile =\> 0 / 1 / 2 / 3 / 4

We concatenated an augmented metric called BMI, it is stated by domain experts "doctors" that it is an effective way to measure the healthy state of an individual.

### 1. Correlation Matrix

### ![correlation Matrix](https://github.com/abeerhbadr/ML_Project-Body_Level_Classification/assets/56984371/7764bb1c-f93f-428c-8443-e2c894ec5973)

**Notable few:**
| **Col** | Value |
| --- | --- |
| ('BMI', 'Weight') | 0.93 |
| ('Body\_Level', 'BMI') | 0.90 |
| ('Body\_Level', 'Weight') | 0.84 |
| ('Transport', 'Age') | 0.58 |
| ('Body\_Level', 'Fam\_Hist') | 0.50 | 
| ('Fam\_Hist', 'Weight') | 0.49 |
| ('Body\_Level', 'Age') | 0.33 |

### **2. Columns Values Histogram**

### ![hist](https://github.com/abeerhbadr/ML_Project-Body_Level_Classification/assets/56984371/ecfe723b-ce63-46cf-baa2-23f123b271f3)

### **3. Class Frequency (Prior)**
| Body Level | Frequency |
| Body Level 4 | 680 |
| Body Level 3 | 406 |
| Body Level 2  | 201 |
| Body Level 1 | 190 |

We can see that imbalance exists but not very extreme, we will tackle solutions in preprocessing.

### **4. Features distributions among classes** (x-axis bounds are fixed)

### ![classes](https://github.com/abeerhbadr/ML_Project-Body_Level_Classification/assets/56984371/3f006652-e0a6-43f4-b43e-4e9002375f16)

![classes2](https://github.com/abeerhbadr/ML_Project-Body_Level_Classification/assets/56984371/3fb85389-e4e0-4262-93ee-cbfcd7b961a9)



We can observe that most correlation with Body\_Level falls into three categories:

- Highly Positively correlated:

It is most clear in BMI but can be seen in Weight, Age, Family\_Hist, …

- Positively correlated but not too high:

In Alcohol Consump, H Cal Consump. and Water consump. and others

- Negatively correlated:

Like Time\_E\_Dev and Phy\_act it decreases with higher class

- Uncorrelated or have very little influence:

Like Veg Consumption or Smoking

## II. Data Preprocessing
### Data Encoding

- Categorical columns have the following values:

| **Column Name** | **Unique Values** |
| --- | --- |
| Gender | Female - Male |
| H\_Cal\_Consump | No - Yes |
| Smoking | No - Yes |
| Fam\_Hist | No - Yes |
| H\_Cal\_Burn | No - Yes |
| Alcohol\_Consump | No - Sometimes - Frequently - Always |
| Food\_Between\_Meals | No - Sometimes - Frequently - Always |
| Transport | Bike - Walking - Public Transportation - Motor-Bike - AutoMobile |
| Body\_Level | Body\_Level 1, Body\_Level 2, Body\_Level 3, Body\_Level 4 |

We notice that there is an order in each of the columns Alcohol\_Consump, Food\_Between\_Meals, and Body\_Level. So, we map the values as follows:

No = 0, Sometimes=1, Frequently=2, Always=3

No = 0, Yes = 1

Body\_Level{i} = i-1

We called this LabelOrdinalEncode

We also tried One-Hot encoding of categorical features, which makes all values in the whole dataframe 0/1, for example: Alcohol\_Consump above becomes 4 columns. Alcohol\_Consump\_no, Alcohol\_Consump\_sometimes, Alcohol\_Consump\_Frequently, Alcohol\_Consump\_Always, and only one of them is 1 in each row depending on the value of the original row.

1.
### Data Resampling

- We have tried two methods for upsampling:

1. sklearn's resample which does simple random sampling with replacement. This means all rows in the original dataset have equal probabilities of being chosen to fill minority class
2. SMOTE, which looks for the nearest k neighbours for a sample, chooses one neighbour at random, then adds a value between the two samples' vectors

- We found that not resampling yields higher accuracies.
- 
### Aggregates

From [Correlation Matrix](#_xpi09cb1zjhj), we find high correlation between Body\_Level and weight, while we find quite a low correlation between Body\_Level and Height. So we add aggregate columns that have meaning combining two features or more. The aggregate column we added is the Body Mass Index (BMI).

- BMI is an indicator of how healthy a person is. Its equation is
- If BMI \<18.5, The person is underweighted
- If The person is normal
- If The Person is overweight
- If The person has obesity.

This addition of BMI column added a significant improvement to the accuracy.

## III. Model/Classifier training

### SVM

- Support vector machine
- Produce high accuracy with less computation power
- Used for both regression and classification.(more common in classification)
- Its main goal is to maximize the margin between the data points and the hyperplane.
- For non-linearly separable data, SVM uses the kernel trick to transform the data into higher dimensionals where it becomes linearly separable
- HyperParameter Tuning Results: {'C': 12, 'gamma': 0.01, 'kernel': 'rbf'}

![0_9jEWNXTAao7phK-5](https://github.com/abeerhbadr/ML_Project-Body_Level_Classification/assets/56984371/957a7978-74ad-4d92-8825-efc402743de0)

![0_0o8xIA4k3gXUDCFU](https://github.com/abeerhbadr/ML_Project-Body_Level_Classification/assets/56984371/04d62c28-b6a0-4dae-a687-d839aeded5f2)

- SVM Learning Curve: used to determine the training set size
![curve training](https://github.com/abeerhbadr/ML_Project-Body_Level_Classification/assets/56984371/19c5737d-65df-4305-9150-e1d126e16f1e)

- SVM Validation Curve of the "gamma" parameter:

![svm validation](https://github.com/abeerhbadr/ML_Project-Body_Level_Classification/assets/56984371/b0d36612-bdd1-46b2-a1c7-d98b0c845c81)


- SVM Validation Curve of the "C" parameter:

![validation2](https://github.com/abeerhbadr/ML_Project-Body_Level_Classification/assets/56984371/60ae3568-992a-4e6c-9ff5-be71e0fe3745)


### **Logistic Regression**
  - Assumes that the relationship between the input variables and the output variable is linear
  - Assumes that the input variables are independent of each other
  - uses a logistic function, also known as a sigmoid function
  - sigmoid(z) = 1 / (1 + exp(-z))
  - z = β0 + β1x1 + β2x2 + ... + βpxp where x is input and β are weights
  - We set the parameter penalty='None'
**Since we have 4 classes for human body level we used Multinomial logistic regression**

### Adaboost
  - boosting → ensemble modeling technique that attempts to build a strong classifier for numbers of weak classifiers.
  - It is done by building a model by using weak models in series.
  - AdaBoost → the first really successful boosting algorithm developed for the purpose of binary classification. 
  - AdaBoost is short for Adaptive Boosting and is a very popular boosting technique that combines multiple "weak classifiers" into a single "strong classifier"
  - Algorithm:
    1. Initialize the dataset and assign equal weight to each of the data points.
    2. Provide this as input to the model and identify the wrongly classified data points.
    3. Increase the weight of the wrongly classified data points.
    4. if (got required results): 
        Goto step 5
      else: 
        Goto step 2
    5. End

![adaboost](https://github.com/abeerhbadr/ML_Project-Body_Level_Classification/assets/56984371/0c434c35-7c2e-49c6-a616-b0abb9e9f1ce)

![adaboost2](https://github.com/abeerhbadr/ML_Project-Body_Level_Classification/assets/56984371/cb099501-0ff3-4598-af34-e637c902c046)

Adaboost Learning Curve:

![adaboost_learning curve](https://github.com/abeerhbadr/ML_Project-Body_Level_Classification/assets/56984371/3f929fe2-c75c-4ddb-9494-66667b91146a)

Adaboost Validation Curve of n\_estimators:

![adaboostwithn_estimator](https://github.com/abeerhbadr/ML_Project-Body_Level_Classification/assets/56984371/ca523a0d-ece0-492e-8b56-e61b4caed81c)

### XGBOOST
  - Extreme Gradient Boosting
  - is a decision-tree-based ensemble Machine Learning algorithm that uses a [_gradient  boosting_](https://en.wikipedia.org/wiki/Gradient_boosting) framework.
  - prediction problems involving unstructured data (images, text, etc.) artificial neural networks tend to outperform all other algorithms or frameworks.
 
![xgboost](https://github.com/abeerhbadr/ML_Project-Body_Level_Classification/assets/56984371/223cfb6e-b2ad-4fdd-abdd-c2483f52d69b)

- Algorithm
  - It is an ensemble learning method that combines the predictions of multiple weak models to produce a stronger prediction.
  - operates on decision trees, models that construct a graph that examines the input under various "if" statements.
    - Whether the "if" condition is satisfied influences the next "if" condition and eventual prediction.
    - 
XGBoost Validation Curve with n\_estimators:

![xgboost curve](https://github.com/abeerhbadr/ML_Project-Body_Level_Classification/assets/56984371/997aab14-f3b7-4ae5-91d1-eb2552fb54df)

**Bias-Variance Analysis in Models** (results in next section)

Bias-variance analysis uses the decomposition of _E__out_ into two distinct components to easily tackle and analyze models and hypothesis sets.

Bias refers to the error introduced by using a model/hypothesis set that is too simple for the true function we are estimating.

Variance refers to the error that is introduced by the model's sensitivity to small fluctuations in the training data, it gets higher when the model is too complex for the problem that it's starting to fit the noise introduced in the sample data it sees (overfit)

![variance](https://github.com/abeerhbadr/ML_Project-Body_Level_Classification/assets/56984371/aa655159-3874-4634-8d4f-248cbad43d01)

## IV. Results and Evaluation

1. Preprocessing used: ([Aggregate](#_sff8mfl4rbop), [LabelOrdinalEncode](#_q1aar8xo4p3o), No Replacing)

- Models metrics on Training data

| Model | Precision % | Recall % | F1 % | Accuracy % |
| --- | --- | --- | --- | --- |
| MultinomialNB | 68.09 | 69.09 | 67.33 | 69.09 |
| Linear Regression | 86.05 | 84.08 | 84.60 | 84.08 |
| LogisticRegression | 99.66 | 99.66 | 99.66 | 99.66 |
| SVM | 99.92 | 99.92 | 99.92 | 99.92 |
| SVM (RBF) | 85.43 | 83.66 | 82.91 | 83.66 |
| AdaBoost | 98.79 | 98.73 | 98.74 | 98.73 |
| CatBoost | 99.75 | 99.75 | 99.75 | 99.75 |
| XGBoost | 100 | 100 | 100 | 100 |

- Models metrics on Validation data

| **Model** | **Precision%** | **Recall** % | **Accuracy %** | **F1** % | **Bias** | **Variance** |
| --- | --- | --- | --- | --- | --- | --- |
| MultinomialNB | 64.30 | 64.53 | 64.53 | 62.25 | 0.348 | 0.0513 |
| Linear Regression | 84.67 | 82.09 | 82.09 | 82.64 | 0.463 | 0.033 |
| LogisticRegression | 98.36 | 98.31 | 98.31 | 98.29 | 0.020 | 0.0159 |
| SVM | 97.00 | 96.96 | 96.96 | 96.94 | 0.024 | 0.0188 |
| SVM (RBF) | 85.74 | 82.43 | 82.43 | 82.08 | 0.169 | 0.0216 |
| RandomForest | 99.33 | 99.32 | 99.32 | 99.32 | – | – |
| AdaBoost | 99.66 | 99.66 | 99.66 | 99.66 | 0.003 | 0.0697 |
| CatBoost | 100 | 100 | 100 | 100 | — | — |
| XGBoost | 100 | 100 | 100 | 100 | 0.000 | 0.0047 |
| LightGBM | 99.32 | 99.32 | 99.32 | 99.32 | — | — |

1. Preprocessing used: OneHot Encoding

- Models metrics on [Test]

| Model | Precision % | Recall % | F1 % | Accuracy % |
| --- | --- | --- | --- | --- |
| MultinomialNB | 67.59 | 67.91 | 67.28 | 67.91 |
| Linear Regression | 83.53 | 80.74 | 81.34 | 80.74 |
| LogisticRegression | 90.15 | 89.86 | 89.77 | 89.86 |
| SVM | 98.00 | 97.97 | 97.98 | 97.97 |
| SVM (RBF) | 76.68 | 74.66 | 74.27 | 74.66 |
| AdaBoost | 80.92 | 65.20 | 64.06 | 65.20 |
| CatBoost | 96.48 | 96.28 | 96.17 | 96.28 |
| XGBoost | 97.32 | 97.30 | 97.27 | 97.30 |

1. Preprocessing used: (No Aggregates, LabelOrdinalEncoding, No Resampling)

- Models metrics on [Test]

| Model | Precision % | Recall % | F1 % | Accuracy % |
| --- | --- | --- | --- | --- |
| MultinomialNB | 65.66 | 66.89 | 64.40 | 66.89 |
| Linear Regression | 82.42 | 79.05 | 79.70 | 79.05 |
| LogisticRegression | 88.50 | 88.51 | 88.15 | 88.51 |
| SVM | 98.03 | 97.97 | 97.97 | 97.97 |
| SVM (RBF) | 76.68 | 74.66 | 74.27 | 74.66 |
| AdaBoost | 63.16 | 64.53 | 61.83 | 64.53 |
| CatBoost | 95.76 | 95.61 | 95.47 | 95.61 |
| XGBoost | 97.36 | 97.30 | 97.27 | 97.30 |

1. Preprocessing used: (No Aggregates, LabelOrdinalEncoding, SMOTE)

- Models metrics on [Test]

| Model | Precision % | Recall % | F1 % | Accuracy % |
| --- | --- | --- | --- | --- |
| MultinomialNB | 70.41 | 68.24 | 68.73 | 68.24 |
| Linear Regression | 85.11 | 81.42 | 82.49 | 81.42 |
| LogisticRegression | 86.94 | 86.49 | 86.60 | 86.49 |
| SVM | 98.06 | 97.97 | 97.99 | 97.97 |
| SVM (RBF) | 77.11 | 73.99 | 74.69 | 73.99 |
| AdaBoost | 77.00 | 70.27 | 68.19 | 70.27 |
| CatBoost | 96.71 | 96.92 | 96.54 | 96.62 |
| XGBoost | 96.31 | 96.28 | 96.21 | 96.28 |

1. Preprocessing used: (No Aggregates, LabelOrdinalEncoding, Random Resampling with replacement)

- Models metrics on [Test]

| Model | Precision % | Recall % | F1 % | Accuracy % |
| --- | --- | --- | --- | --- |
| MultinomialNB | 72.82 | 70.95 | 70.96 | 70.95 |
| Linear Regression | 86.41 | 82.77 | 83.78 | 82.77 |
| LogisticRegression | 85.23 | 84.80 | 84.91 | 84.80 |
| SVM | 98.05 | 97.97 | 97.97 | 97.97 |
| SVM (RBF) | 76.86 | 73.65 | 74.41 | 73.65 |
| AdaBoost | 79.86 | 67.57 | 66.53 | 67.57 |
| CatBoost | 97.71 | 97.64 | 97.60 | 97.64 |
| XGBoost | 96.27 | 96.28 | 96.26 | 96.28 |

1. Preprocessing used: (Aggregates (BMI column), LabelOrdinalEncoding, Random Resampling with replacement)

- Models metrics on [Test]

| Model | Precision % | Recall % | F1 % | Accuracy % |
| --- | --- | --- | --- | --- |
| MultinomialNB | 72.88 | 71.62 | 71.82 | 71.62 |
| Linear Regression | 87.20 | 84.80 | 85.19 | 84.80 |
| LogisticRegression | 97.05 | 96.96 | 96.95 | 96.96 |
| SVM | 96.99 | 96.96 | 96.93 | 96.96 |
| SVM (RBF) | 88.66 | 85.81 | 85.95 | 85.81 |
| AdaBoost | 98.41 | 98.31 | 98.32 | 98.31 |
| CatBoost | 100 | 100 | 100 | 100 |
| XGBoost | 100 | 100 | 100 | 100 |

## V. Contribution of team members

- Gheiath Omar
  - Did Data Visualization and Analysis
  - Contributed two ideas in preprocessing (aggregate + ordinal label encoding)
  - Contributed in Model's testing and parameter tuning
  - Bias Variance Analysis

- Nour-AlDin
  - Contributed in Data Visualization and Analysis
  - Contributed in Model's testing and parameter tuning
  - Bias Variance Analysis

- Abeer Hussein
  - Simple Analysis
  - Cross Validation, drawing learning curves for train and validations sets sizes
  - Validation Curves for the different models parameters
  - hyperparameter Tuning

- Mohamed Akram
  - Contributed in Data Visualization and Analysis
  - Contributed to Cross Validation for models
  - Tried preprocessing techniques
    - One hot encoding, label ordinal encoding
    - Smote resampling, random resampling, no resampling
    - Adding aggregate columns
  - Tried training on subsets of features
