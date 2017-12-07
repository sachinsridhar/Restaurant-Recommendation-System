---
title: Baseline Model
notebook: Modeling.ipynb
nav_include: 2
---

## Contents
{:.no_toc}
*  
{: toc}


## Data Wrangling

Data has already been filtered based on geographical location (only restaurants in two states of Canada - Ontario and Quebec - have been selected) and basic data cleaning operations have been carried out on the datasets shared by Yelp.

Based on the results of the EDA and the design decisions made for the analysis, we perform data wrangling, the steps of which have been outlined below -
1. Eliminate restaurants with a low number of reviews (for reliability of information)
2. Eliminate users with a low number of restaurant reviews
3. Perform one-hot encoding for users and restaurants
4. Split the reviews into three datasets randomly: training, test, and holdout, such that every restaurant is in the training set and that every user is in each dataset.

After applying the filters, we see below the dimensions of the dataset that we will perform our analysis on.

    # users with more than 50 reviews (review2):  1073
    total # users (review2):                      1073
    total # restaurants (review2):                7345
    total # reviews (review2):                    103112

The next task is to split the data, and we start by permuting the data. The objective is to make sure that each user is present in every split and that the training set contains every restaurant in review2. This must be done to ensure that we have some information about each user and that the matrix we factorize covers all the possibilities. The resulting size of each split is displayed below, as well as a sample table of occurrences of users in each split.

    size of training set:  21154
    size of test set:      14331
    size of holdout set:   67627


<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>training</th>
      <th>test</th>
      <th>holdout</th>
    </tr>
    <tr>
      <th>user_id</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>--Qh8yKWAvIP4V4K8ZPfHA</th>
      <td>38</td>
      <td>27</td>
      <td>124</td>
    </tr>
    <tr>
      <th>-B4Cf2XLkPr9qMlLPHJAlw</th>
      <td>12</td>
      <td>9</td>
      <td>45</td>
    </tr>
    <tr>
      <th>-KVxkJDSTjtPGsamMDG92Q</th>
      <td>17</td>
      <td>11</td>
      <td>52</td>
    </tr>
    <tr>
      <th>-KpEgEen1tj-jdjIS7uVOw</th>
      <td>23</td>
      <td>8</td>
      <td>37</td>
    </tr>
    <tr>
      <th>-RCD8F7qbsLfzT3k1HtMxg</th>
      <td>12</td>
      <td>9</td>
      <td>43</td>
    </tr>
  </tbody>
</table>
</div>



We make two observations from the outputs displayed above. First, we notice the split between the training, test and the hold-out datasets. Second, in the table displayed above, we find a description of how the reviews of each user have been split across the three datasets. It has been ensured that the reviews of a user are well represented in all the three datasets - the training, test and hold-out. This is again comfirmed in the output below.


    number of unique users:                            1073
    number of unique users in the training set:        1073
    number of unique users in the test set:            1073
    number of unique users in the holdout set:         1073
    number of unique restaurants:                      7345
    number of unique restaurants in the training set:  7345


A sample of the binarized training data set to be used in the baseline regression is displayed below.


<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id_-B4Cf2XLkPr9qMlLPHJAlw</th>
      <th>user_id_-KVxkJDSTjtPGsamMDG92Q</th>
      <th>user_id_-KpEgEen1tj-jdjIS7uVOw</th>
      <th>user_id_-RCD8F7qbsLfzT3k1HtMxg</th>
      <th>user_id_-_2h2cJlBOWAYrfplMU-Cg</th>
      <th>user_id_-d2daWmftYumOaYpbD5D8Q</th>
      <th>user_id_-dbWm5L_Ol2hZeLRoQOK7w</th>
      <th>user_id_-fEe8XBeJ6pGLIeAyAWzfw</th>
      <th>user_id_-hUgrj7Lzir3yLUYrMYQ4g</th>
      <th>user_id_-m0KTRk0c901-4b-BN34Gg</th>
      <th>...</th>
      <th>business_id_zvtkeghW0Px5HY9QkJ4INw</th>
      <th>business_id_zw4Legbcu018p5WcZ74iWA</th>
      <th>business_id_zw74kL1IvT65yRvNLx5UxA</th>
      <th>business_id_zwkif4XLEDqdEwEgTWLIVQ</th>
      <th>business_id_zxJlg4XCHNoFy78WZPv89w</th>
      <th>business_id_zy_NHTqtfSrfTGGPoqy4Mw</th>
      <th>business_id_zyw5DjrRks7a8OhmBsgCQQ</th>
      <th>business_id_zz3CqZhNx2rQ_Yp6zHze-A</th>
      <th>business_id_zze6IysT7bJFS8gvi6fZ2A</th>
      <th>business_id_zzlZJVkEhOzR2tJOLHcF2A</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2721</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>39119</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 8416 columns</p>
</div>



## Basline Linear Regression Model
In the first part of our analysis, we create a baseline estimate of the ratings. We fit the baseline linear regression model, represented mathematically as 
$\hat{Y}_{ur}=\hat{\mu}+\hat{\theta}_u+\hat{\gamma}_r$, 
where $\hat{\theta}_u$ represents the bias of a particular user, and $\hat{\gamma}_r$ the bias of a particular restaurant.



```python
# Fitting the linear baseline regression model
regress1 = LinearRegression()
regress1.fit(train_set_x, train_set_y)
print('test score 1:    ', regress1.score(test_set_x, test_set_y))
```


    test score 1:     -0.130286650544


We notice a poor score for the Test/Validation $R^2$, and hence perform Regulzarized Regression (with Lasso).



```python
lambdas = [.00001, .0001, .001,.005,1,5,10,50,100,500,1000]
regress2 = LassoCV(cv=10, alphas=lambdas)
regress2.fit(train_set_x, train_set_y)
print('test score 2:    ', regress2.score(test_set_x, test_set_y))
```


    test score 2:     0.0993635027725


In contrast to the Linear Regression Model without Regularization, we observe a significant improvement in the $R^2$ value on the test dataset. As a consequence of improved performance of the Lasso Regularized Regression Model, we choose this as our baseline, and perform all subsequent checks/comparisons with respect to this model. We use the residuals obtained from the regression model in the next stage of our analysis.
