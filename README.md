# Prediction of Credit Card fraud
- The [creditcard.csv](https://kh3-ls-storage.s3.us-east-1.amazonaws.com/Updated%20Project%20guide%20data%20set/creditcard.csv) contains a reasonable large number of data related with credic card transactions.

## Problem Statement:
A credit card is one of the most used financial products to make online purchases and payments. Though the Credit cards can be a convenient way to manage your finances, they can also be risky. Credit card fraud is the unauthorized use of someone else's credit card or credit card information to make purchases or withdraw cash.
It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase. 
The dataset contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
We have to build a classification model to predict whether a transaction is fraudulent or not.




## Recommended Steps
1. **Exploratory Data Analysis:** Analyze and understand the data to identify patterns, relationships, and trends in the data by using Descriptive Statistics and Visualizations. 
1. **Data Cleaning:** This might include standardization, handling the missing values and outliers in the data.
1. **Dealing with Imbalanced data:** This data set is highly imbalanced. The data should be balanced using the appropriate  methods before moving onto model building.
1. **Feature Engineering:** Create new features or transform the existing features for better performance of the ML Models. 
1. **Model Selection:** Choose the most appropriate model that can be used for this project. 
1. **Model training:** Split the data into train & test sets and use the train set to estimate the best model parameters. 
1. **Model Validation:** Evaluate the performance of the model on data that was not used during the training process. The goal is to estimate the model's ability to generalize to new, unseen data and to identify any issues with the model, such as overfitting.
1. **Model Deployment:** Model deployment is the process of making a trained machine learning model available for use in a production environment. 



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```


```python
# loading the dataset
data = pd.read_csv("https://kh3-ls-storage.s3.us-east-1.amazonaws.com/Updated%20Project%20guide%20data%20set/creditcard.csv")
# first 5 rows of the dataset
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>-1.359807</td>
      <td>-0.072781</td>
      <td>2.536347</td>
      <td>1.378155</td>
      <td>-0.338321</td>
      <td>0.462388</td>
      <td>0.239599</td>
      <td>0.098698</td>
      <td>0.363787</td>
      <td>...</td>
      <td>-0.018307</td>
      <td>0.277838</td>
      <td>-0.110474</td>
      <td>0.066928</td>
      <td>0.128539</td>
      <td>-0.189115</td>
      <td>0.133558</td>
      <td>-0.021053</td>
      <td>149.62</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.191857</td>
      <td>0.266151</td>
      <td>0.166480</td>
      <td>0.448154</td>
      <td>0.060018</td>
      <td>-0.082361</td>
      <td>-0.078803</td>
      <td>0.085102</td>
      <td>-0.255425</td>
      <td>...</td>
      <td>-0.225775</td>
      <td>-0.638672</td>
      <td>0.101288</td>
      <td>-0.339846</td>
      <td>0.167170</td>
      <td>0.125895</td>
      <td>-0.008983</td>
      <td>0.014724</td>
      <td>2.69</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>-1.358354</td>
      <td>-1.340163</td>
      <td>1.773209</td>
      <td>0.379780</td>
      <td>-0.503198</td>
      <td>1.800499</td>
      <td>0.791461</td>
      <td>0.247676</td>
      <td>-1.514654</td>
      <td>...</td>
      <td>0.247998</td>
      <td>0.771679</td>
      <td>0.909412</td>
      <td>-0.689281</td>
      <td>-0.327642</td>
      <td>-0.139097</td>
      <td>-0.055353</td>
      <td>-0.059752</td>
      <td>378.66</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>-0.966272</td>
      <td>-0.185226</td>
      <td>1.792993</td>
      <td>-0.863291</td>
      <td>-0.010309</td>
      <td>1.247203</td>
      <td>0.237609</td>
      <td>0.377436</td>
      <td>-1.387024</td>
      <td>...</td>
      <td>-0.108300</td>
      <td>0.005274</td>
      <td>-0.190321</td>
      <td>-1.175575</td>
      <td>0.647376</td>
      <td>-0.221929</td>
      <td>0.062723</td>
      <td>0.061458</td>
      <td>123.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>-1.158233</td>
      <td>0.877737</td>
      <td>1.548718</td>
      <td>0.403034</td>
      <td>-0.407193</td>
      <td>0.095921</td>
      <td>0.592941</td>
      <td>-0.270533</td>
      <td>0.817739</td>
      <td>...</td>
      <td>-0.009431</td>
      <td>0.798278</td>
      <td>-0.137458</td>
      <td>0.141267</td>
      <td>-0.206010</td>
      <td>0.502292</td>
      <td>0.219422</td>
      <td>0.215153</td>
      <td>69.99</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>



# Data exploration and cleaning


```python
# dataset information
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 284807 entries, 0 to 284806
    Data columns (total 31 columns):
     #   Column  Non-Null Count   Dtype  
    ---  ------  --------------   -----  
     0   Time    284807 non-null  float64
     1   V1      284807 non-null  float64
     2   V2      284807 non-null  float64
     3   V3      284807 non-null  float64
     4   V4      284807 non-null  float64
     5   V5      284807 non-null  float64
     6   V6      284807 non-null  float64
     7   V7      284807 non-null  float64
     8   V8      284807 non-null  float64
     9   V9      284807 non-null  float64
     10  V10     284807 non-null  float64
     11  V11     284807 non-null  float64
     12  V12     284807 non-null  float64
     13  V13     284807 non-null  float64
     14  V14     284807 non-null  float64
     15  V15     284807 non-null  float64
     16  V16     284807 non-null  float64
     17  V17     284807 non-null  float64
     18  V18     284807 non-null  float64
     19  V19     284807 non-null  float64
     20  V20     284807 non-null  float64
     21  V21     284807 non-null  float64
     22  V22     284807 non-null  float64
     23  V23     284807 non-null  float64
     24  V24     284807 non-null  float64
     25  V25     284807 non-null  float64
     26  V26     284807 non-null  float64
     27  V27     284807 non-null  float64
     28  V28     284807 non-null  float64
     29  Amount  284807 non-null  float64
     30  Class   284807 non-null  int64  
    dtypes: float64(30), int64(1)
    memory usage: 67.4 MB
    


```python
# checking the missing values in each column
data.isnull().sum()
```




    Time      0
    V1        0
    V2        0
    V3        0
    V4        0
    V5        0
    V6        0
    V7        0
    V8        0
    V9        0
    V10       0
    V11       0
    V12       0
    V13       0
    V14       0
    V15       0
    V16       0
    V17       0
    V18       0
    V19       0
    V20       0
    V21       0
    V22       0
    V23       0
    V24       0
    V25       0
    V26       0
    V27       0
    V28       0
    Amount    0
    Class     0
    dtype: int64




```python
# checking for fraudulent transactions
data.Class.value_counts()
```




    0    284315
    1       492
    Name: Class, dtype: int64




```python
# separating the data for analysis
legit = data[data.Class == 0]
fraud = data[data.Class == 1]
print(legit.shape)
print(fraud.shape)
```

    (284315, 31)
    (492, 31)
    


```python
# statistical measures of the data
legit.Amount.describe()
```




    count    284315.000000
    mean         88.291022
    std         250.105092
    min           0.000000
    25%           5.650000
    50%          22.000000
    75%          77.050000
    max       25691.160000
    Name: Amount, dtype: float64




```python
fraud.Amount.describe()
```




    count     492.000000
    mean      122.211321
    std       256.683288
    min         0.000000
    25%         1.000000
    50%         9.250000
    75%       105.890000
    max      2125.870000
    Name: Amount, dtype: float64




```python
# compare the values for both transactions
data.groupby('Class').mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V20</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
    </tr>
    <tr>
      <th>Class</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>94838.202258</td>
      <td>0.008258</td>
      <td>-0.006271</td>
      <td>0.012171</td>
      <td>-0.007860</td>
      <td>0.005453</td>
      <td>0.002419</td>
      <td>0.009637</td>
      <td>-0.000987</td>
      <td>0.004467</td>
      <td>...</td>
      <td>-0.000644</td>
      <td>-0.001235</td>
      <td>-0.000024</td>
      <td>0.000070</td>
      <td>0.000182</td>
      <td>-0.000072</td>
      <td>-0.000089</td>
      <td>-0.000295</td>
      <td>-0.000131</td>
      <td>88.291022</td>
    </tr>
    <tr>
      <th>1</th>
      <td>80746.806911</td>
      <td>-4.771948</td>
      <td>3.623778</td>
      <td>-7.033281</td>
      <td>4.542029</td>
      <td>-3.151225</td>
      <td>-1.397737</td>
      <td>-5.568731</td>
      <td>0.570636</td>
      <td>-2.581123</td>
      <td>...</td>
      <td>0.372319</td>
      <td>0.713588</td>
      <td>0.014049</td>
      <td>-0.040308</td>
      <td>-0.105130</td>
      <td>0.041449</td>
      <td>0.051648</td>
      <td>0.170575</td>
      <td>0.075667</td>
      <td>122.211321</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 30 columns</p>
</div>




```python
# utilities for plotting data
def countplot_data(data, feature):
    '''
        Method to compute countplot of given dataframe
        Parameters:
            data(pd.Dataframe): Input Dataframe
            feature(str): Feature in Dataframe
    '''
    plt.figure(figsize=(10,10))
    sns.countplot(x=feature, data=data)
    plt.show()

def pairplot_data_grid(data, feature1, feature2, target):
    '''
        Method to construct pairplot of the given feature wrt data
        Parameters:
            data(pd.DataFrame): Input Dataframe
            feature1(str): First Feature for Pair Plot
            feature2(str): Second Feature for Pair Plot
            target: Target or Label (y)
    '''

    sns.FacetGrid(data, hue=target).map(plt.scatter, feature1, feature2).add_legend()
    plt.show()

```


```python
# unbalanced data with legitimate and fraudulent data as count plot
countplot_data(data, data.Class)
```


    
![png](output_12_0.png)
    



```python
# relationship between fraud transactions and amount of money
pairplot_data_grid(data, "Time", "Amount", "Class")
```


    
![png](output_13_0.png)
    



```python
pairplot_data_grid(data, "Amount", "Time", "Class")
```


    
![png](output_14_0.png)
    


* The Data does not have any missing values and hence, need not be handled.
* The Data has only Target Variable Class as the categorical variable.
* Remaining Features are numerical and need to be only standardized for comparison after balancing the dataset
* The mean of the amount of money in transactions is 88.34
* The standard deviation of amount of money in transactions is 250.12
* The time is distributed throughout the data equitably and hence, serves as an independent feature
* It is best to not remove or drop any data or features in this case and try to tune the model assuming them as independent features initially
* The Dataset has 31 columns with unknown features labelled V1 to V28, Time, Amount and Class
* The target variable is 'Class' and rest of the variables are input features
* The Class has the following values:
  0: Legitimate Transactions
  1: Fraud Transactions
* It can be observed that the fraud transactions are generally not above an amount of 2500.
* It can also be observed that the fraud transactions are evenly distributed about time.
* The Dataset is highly imbalanced as evident from the countplot with majoritarian class label '0' and minority class label '1'. Thus, if we run the model on such imbalanced data we may end up highly overfitting it on the data and resulting in non-deployable model. Hence, we will perform Synthetic Minority Oversampling on the data to balance it out as shown later after exploring other features.


```python
# Build a sample dataset containing similar distribution of normal transactions and Fraudulent Transactions
legit_sample = legit.sample(n=492)
new_dataset = pd.concat([legit_sample, fraud], axis=0)
new_dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>87408</th>
      <td>61674.0</td>
      <td>1.175464</td>
      <td>-0.207412</td>
      <td>1.130632</td>
      <td>0.665067</td>
      <td>-0.875796</td>
      <td>0.206954</td>
      <td>-0.753863</td>
      <td>0.251648</td>
      <td>0.716198</td>
      <td>...</td>
      <td>-0.067888</td>
      <td>-0.027160</td>
      <td>-0.014207</td>
      <td>0.042001</td>
      <td>0.266692</td>
      <td>0.320851</td>
      <td>0.013024</td>
      <td>0.012253</td>
      <td>1.94</td>
      <td>0</td>
    </tr>
    <tr>
      <th>218992</th>
      <td>141551.0</td>
      <td>2.287804</td>
      <td>-0.596634</td>
      <td>-1.757361</td>
      <td>-1.170007</td>
      <td>0.136674</td>
      <td>-0.556089</td>
      <td>-0.268966</td>
      <td>-0.380078</td>
      <td>-0.729338</td>
      <td>...</td>
      <td>0.396290</td>
      <td>1.206090</td>
      <td>-0.089671</td>
      <td>0.134978</td>
      <td>0.375195</td>
      <td>0.073525</td>
      <td>-0.034984</td>
      <td>-0.062598</td>
      <td>15.00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>80957</th>
      <td>58735.0</td>
      <td>-3.314928</td>
      <td>-2.984825</td>
      <td>0.916308</td>
      <td>-0.948370</td>
      <td>1.810882</td>
      <td>-1.330976</td>
      <td>-1.491131</td>
      <td>0.875191</td>
      <td>-1.598434</td>
      <td>...</td>
      <td>0.644281</td>
      <td>0.505317</td>
      <td>-0.286193</td>
      <td>-0.232554</td>
      <td>0.600674</td>
      <td>-0.136332</td>
      <td>-0.051825</td>
      <td>-0.521684</td>
      <td>50.00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>30964</th>
      <td>36156.0</td>
      <td>1.295712</td>
      <td>0.271130</td>
      <td>0.314291</td>
      <td>0.499473</td>
      <td>-0.195799</td>
      <td>-0.696756</td>
      <td>0.083251</td>
      <td>-0.200471</td>
      <td>0.007887</td>
      <td>...</td>
      <td>-0.284781</td>
      <td>-0.797589</td>
      <td>0.069250</td>
      <td>-0.107037</td>
      <td>0.299867</td>
      <td>0.123912</td>
      <td>-0.026631</td>
      <td>0.014978</td>
      <td>1.79</td>
      <td>0</td>
    </tr>
    <tr>
      <th>162259</th>
      <td>114968.0</td>
      <td>1.841357</td>
      <td>-0.544824</td>
      <td>-0.301692</td>
      <td>0.403789</td>
      <td>-0.788328</td>
      <td>-0.717143</td>
      <td>-0.335490</td>
      <td>-0.170293</td>
      <td>0.928977</td>
      <td>...</td>
      <td>-0.154560</td>
      <td>-0.464457</td>
      <td>0.339302</td>
      <td>0.073006</td>
      <td>-0.635255</td>
      <td>0.258801</td>
      <td>-0.043419</td>
      <td>-0.026096</td>
      <td>90.00</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>




```python
new_dataset['Class'].value_counts()
```




    0    492
    1    492
    Name: Class, dtype: int64




```python
new_dataset.groupby('Class').mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V20</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
    </tr>
    <tr>
      <th>Class</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>93845.130081</td>
      <td>-0.231761</td>
      <td>-0.017296</td>
      <td>-0.007106</td>
      <td>-0.051681</td>
      <td>-0.036262</td>
      <td>-0.048506</td>
      <td>-0.036959</td>
      <td>0.003817</td>
      <td>0.060338</td>
      <td>...</td>
      <td>0.002928</td>
      <td>-0.025555</td>
      <td>-0.072839</td>
      <td>0.017866</td>
      <td>0.009543</td>
      <td>0.019696</td>
      <td>0.007208</td>
      <td>0.033355</td>
      <td>-0.019686</td>
      <td>96.233862</td>
    </tr>
    <tr>
      <th>1</th>
      <td>80746.806911</td>
      <td>-4.771948</td>
      <td>3.623778</td>
      <td>-7.033281</td>
      <td>4.542029</td>
      <td>-3.151225</td>
      <td>-1.397737</td>
      <td>-5.568731</td>
      <td>0.570636</td>
      <td>-2.581123</td>
      <td>...</td>
      <td>0.372319</td>
      <td>0.713588</td>
      <td>0.014049</td>
      <td>-0.040308</td>
      <td>-0.105130</td>
      <td>0.041449</td>
      <td>0.051648</td>
      <td>0.170575</td>
      <td>0.075667</td>
      <td>122.211321</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 30 columns</p>
</div>



# Feature engineering


```python
# Split data into feature and target
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

print(X)
```

                Time        V1        V2        V3        V4        V5        V6  \
    87408    61674.0  1.175464 -0.207412  1.130632  0.665067 -0.875796  0.206954   
    218992  141551.0  2.287804 -0.596634 -1.757361 -1.170007  0.136674 -0.556089   
    80957    58735.0 -3.314928 -2.984825  0.916308 -0.948370  1.810882 -1.330976   
    30964    36156.0  1.295712  0.271130  0.314291  0.499473 -0.195799 -0.696756   
    162259  114968.0  1.841357 -0.544824 -0.301692  0.403789 -0.788328 -0.717143   
    ...          ...       ...       ...       ...       ...       ...       ...   
    279863  169142.0 -1.927883  1.125653 -4.518331  1.749293 -1.566487 -2.010494   
    280143  169347.0  1.378559  1.289381 -5.004247  1.411850  0.442581 -1.326536   
    280149  169351.0 -0.676143  1.126366 -2.213700  0.468308 -1.120541 -0.003346   
    281144  169966.0 -3.113832  0.585864 -5.399730  1.817092 -0.840618 -2.943548   
    281674  170348.0  1.991976  0.158476 -2.583441  0.408670  1.151147 -0.096695   
    
                  V7        V8        V9  ...       V20       V21       V22  \
    87408  -0.753863  0.251648  0.716198  ... -0.101874 -0.067888 -0.027160   
    218992 -0.268966 -0.380078 -0.729338  ...  0.122365  0.396290  1.206090   
    80957  -1.491131  0.875191 -1.598434  ...  0.840654  0.644281  0.505317   
    30964   0.083251 -0.200471  0.007887  ... -0.046964 -0.284781 -0.797589   
    162259 -0.335490 -0.170293  0.928977  ...  0.040691 -0.154560 -0.464457   
    ...          ...       ...       ...  ...       ...       ...       ...   
    279863 -0.882850  0.697211 -2.064945  ...  1.252967  0.778584 -0.319189   
    280143 -1.413170  0.248525 -1.127396  ...  0.226138  0.370612  0.028234   
    280149 -2.234739  1.210158 -0.652250  ...  0.247968  0.751826  0.834108   
    281144 -2.208002  1.058733 -1.632333  ...  0.306271  0.583276 -0.269209   
    281674  0.223050 -0.068384  0.577829  ... -0.017652 -0.164350 -0.295135   
    
                 V23       V24       V25       V26       V27       V28  Amount  
    87408  -0.014207  0.042001  0.266692  0.320851  0.013024  0.012253    1.94  
    218992 -0.089671  0.134978  0.375195  0.073525 -0.034984 -0.062598   15.00  
    80957  -0.286193 -0.232554  0.600674 -0.136332 -0.051825 -0.521684   50.00  
    30964   0.069250 -0.107037  0.299867  0.123912 -0.026631  0.014978    1.79  
    162259  0.339302  0.073006 -0.635255  0.258801 -0.043419 -0.026096   90.00  
    ...          ...       ...       ...       ...       ...       ...     ...  
    279863  0.639419 -0.294885  0.537503  0.788395  0.292680  0.147968  390.00  
    280143 -0.145640 -0.081049  0.521875  0.739467  0.389152  0.186637    0.76  
    280149  0.190944  0.032070 -0.739695  0.471111  0.385107  0.194361   77.89  
    281144 -0.456108 -0.183659 -0.328168  0.606116  0.884876 -0.253700  245.00  
    281674 -0.072173 -0.450261  0.313267 -0.289617  0.002988 -0.015309   42.53  
    
    [984 rows x 30 columns]
    


```python
print(Y)
```

    87408     0
    218992    0
    80957     0
    30964     0
    162259    0
             ..
    279863    1
    280143    1
    280149    1
    281144    1
    281674    1
    Name: Class, Length: 984, dtype: int64
    


```python
# Split data into training and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
```


```python
print(X.shape, X_train.shape, X_test.shape)
```

    (984, 30) (787, 30) (197, 30)
    

We will use **Logistic regression** model for training and prediction.

# Training models


```python
model = LogisticRegression()
```


```python
# training the Logistic Regression Model with Training Data
model.fit(X_train, Y_train)
```

    C:\Users\swapn\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    




<style>#sk-container-id-3 {color: black;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" checked><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression()</pre></div></div></div></div></div>



# Model Validation


```python
# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
```


```python
print('Accuracy on Training data : ', training_data_accuracy)
```

    Accuracy on Training data :  0.9479034307496823
    


```python
# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
```


```python
print('Accuracy score on Test Data : ', test_data_accuracy)
```

    Accuracy score on Test Data :  0.9390862944162437
    


```python

```
