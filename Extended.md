---
title: Extended Model
notebook: Modeling.ipynb
nav_include: 3
---

## Contents
{:.no_toc}
*  
{: toc}


## Matrix Factorization

The objective is to create a matrix, which contains residuals from the baseline model on the training dataset. The matrix will have the list of users along its rows and the list of restaurants along its columns.

This will be obtained by pivoting our existing dataframe appropriately. For user/restauraunt combinations not in our training set, we set their values to zero, and our goal is to find the *'best guess'* for these values through Matrix Factorization using Alternating Least Squares. By using an appropriate weight matrix $W$ these values will not be interpreted as actual residuals from reviews in the training set.

The resulting matrix of residuals is seen below. Note the sparsity of the matrix: nearly every value is zero.



```python
df_review = train_set_resid.pivot(index = 'user_id', columns ='business_id', values = 'residuals').fillna(0)
df_review.head()
```





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
      <th>business_id</th>
      <th>--6MefnULPED_I942VcFNA</th>
      <th>--DaPTJW3-tB1vP-PfdTEg</th>
      <th>--SrzpvFLwP_YFwB_Cetow</th>
      <th>-0CTrPQNiSyClxhdO4HSDQ</th>
      <th>-0DET7VdEQOJVJ_v6klEug</th>
      <th>-0NhdsDJsdarxyDPR523ZQ</th>
      <th>-0NrB58jqKqJfuUCDupcsw</th>
      <th>-0mm8pqBSIOYZQHeo8XnkA</th>
      <th>-1xuC540Nycht_iWFeJ-dw</th>
      <th>-25X5v1q3WU6s-craJSvTw</th>
      <th>...</th>
      <th>zvtkeghW0Px5HY9QkJ4INw</th>
      <th>zw4Legbcu018p5WcZ74iWA</th>
      <th>zw74kL1IvT65yRvNLx5UxA</th>
      <th>zwkif4XLEDqdEwEgTWLIVQ</th>
      <th>zxJlg4XCHNoFy78WZPv89w</th>
      <th>zy_NHTqtfSrfTGGPoqy4Mw</th>
      <th>zyw5DjrRks7a8OhmBsgCQQ</th>
      <th>zz3CqZhNx2rQ_Yp6zHze-A</th>
      <th>zze6IysT7bJFS8gvi6fZ2A</th>
      <th>zzlZJVkEhOzR2tJOLHcF2A</th>
    </tr>
    <tr>
      <th>user_id</th>
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
      <th>--Qh8yKWAvIP4V4K8ZPfHA</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>-B4Cf2XLkPr9qMlLPHJAlw</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>-KVxkJDSTjtPGsamMDG92Q</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>-KpEgEen1tj-jdjIS7uVOw</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>-RCD8F7qbsLfzT3k1HtMxg</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 7345 columns</p>
</div>



```python
df_review_matrix.shape
```





    (1073, 7345)


We started out with 1073 users and 7345 restaurants in the training dataset, so the dimensions of the matrix seem to be correct.

We setup to apply the Alternating Least Squares Regression method to accomplish matrix factorization. Our objective is to minimize the following loss function:

$\sum_{u,m}(Y_{u,m} - \mu -\bar{\theta}.I_{u} - \bar{\gamma}I_m-\bar{q}_m^T\bar{p}_u)^2 + \alpha(\theta^2 + \gamma_m^2 + \mid\mid\bar{q}_m\mid\mid^2 + \mid\mid\bar{p}_u\mid\mid^2)$

We perform the process of validation in order to tune the parameters alpha (penalty) and the number of latent factors in the matrix. We arrive at the optimal values of *alpha* and number of latent factors to be considered in the process of factorization by computing the sum of squared errors on the test set. The results are shown in the table below.



```python
val_scores
```





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
      <th>lambda</th>
      <th>n_factors</th>
      <th>test/validation sum of squared errors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.05</td>
      <td>4</td>
      <td>24192.03429</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.05</td>
      <td>10</td>
      <td>21338.07873</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.50</td>
      <td>4</td>
      <td>19508.64591</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.50</td>
      <td>10</td>
      <td>16860.38679</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.00</td>
      <td>4</td>
      <td>13330.57669</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5.00</td>
      <td>10</td>
      <td>13275.59727</td>
    </tr>
    <tr>
      <th>6</th>
      <td>50.00</td>
      <td>4</td>
      <td>13215.75947</td>
    </tr>
    <tr>
      <th>7</th>
      <td>50.00</td>
      <td>10</td>
      <td>13215.75947</td>
    </tr>
  </tbody>
</table>
</div>



We then perform matrix factorization on the matrix of residuals using the optimal values for lambda and for the number of latent factors. In the table below we see the residuals for the first two stages of our ensemble model. Note that the residual values predicted from matrix factorization are at or near zero in this sample of observations.





    ********************
    lambda:     50
    n_factors:  4
    iteration 1 is completed
    iteration 2 is completed
    iteration 3 is completed
    iteration 4 is completed
    iteration 5 is completed




```python
train_data = train_set_resid.copy(deep=True)
train_subset = train_data[['business_id','user_id','residuals']]
result_train = pd.merge(train_subset,user_business_residuals, on = ['user_id','business_id'], how = 'inner')
result_train['resid_of_resid'] = result_train['residuals'] - result_train['value']
result_train = result_train.rename(columns={'value': 'residuals_from_mat_factrz', 
                        'residuals': 'residuals_from_linear_regr'})

result_train.head()
```





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
      <th>business_id</th>
      <th>user_id</th>
      <th>residuals_from_linear_regr</th>
      <th>residuals_from_mat_factrz</th>
      <th>resid_of_resid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>bRmb81XDG3E2SOHARBLTog</td>
      <td>oBc0gQ4RpFrqzpNlH6_epA</td>
      <td>0.455356</td>
      <td>9.933523e-17</td>
      <td>0.455356</td>
    </tr>
    <tr>
      <th>1</th>
      <td>X_Pg8SvGGYhCxwWRkrUv3Q</td>
      <td>dT1jqOZrFUmY4m4o37c8rw</td>
      <td>0.279164</td>
      <td>-2.293740e-19</td>
      <td>0.279164</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8xI4hJ3nS4avEoo_l62dkw</td>
      <td>qOdmye8UQdqloVNE059PkQ</td>
      <td>-0.885521</td>
      <td>-1.049530e-17</td>
      <td>-0.885521</td>
    </tr>
    <tr>
      <th>3</th>
      <td>q9_gLvTNf11etVxbH7JY0Q</td>
      <td>Jm5h-bDATqRMWs3VahkFPg</td>
      <td>-0.403760</td>
      <td>-1.918724e-17</td>
      <td>-0.403760</td>
    </tr>
    <tr>
      <th>4</th>
      <td>J9BmILDpV1Pr3GKU9XhjTQ</td>
      <td>Yp7_GeD6KTRoo4Nteqv4SA</td>
      <td>0.392325</td>
      <td>1.714463e-18</td>
      <td>0.392325</td>
    </tr>
  </tbody>
</table>
</div>

The table above steps us through how the results from Matrix Factorization leads us to the next step - the Random Forest model. We have the residuals calculated at the two steps - Baseline Linear Regression and the Matrix Factorization, the difference of which equals the residuals of the calculated residuals, names resid_of_resid in the table above.


## Random Forest Regression

For the final stage of modelling, we sought to create a model that uses additional information about the users or restaurants provided in the dataset. We also engineered several features: day of the week the review was written, month the review was written, length of time the user was active on Yelp when review was written.

We fit an ordinary regression, a ridge regression, and a random forest regression, as some models may be better suited to certain types of problems, for each of the combinations of variables below:
+ Day and month that the review was written, and time user was active on Yelp
+ Restaurant cities as factor variables
+ Latitude and longitude, and their interaction with the state variable

We then perform 10-fold cross validation to assess how well the model would perform out of sample. The output for the cross-validation exercise on the third set of predictors, which proved to be the superior one, is displayed below. The optimal regression technique was the random forest regression.


```python
# simple linear regression
lmdl = LinearRegression()
lmdl.fit(x_train, y_train)
print("10-fold CV R2:", cross_val_score(lmdl, x_train, y_train, cv=10).mean())
```


    10-fold CV R2: -7.91065147508e-05




```python
# ridge regression
rmdl = RidgeCV()
rmdl.fit(x_train, y_train)
print("10-fold CV R2:", cross_val_score(rmdl, x_train, y_train, cv=10).mean())
```


    10-fold CV R2: -7.47857359459e-05




```python
# random forest regression
depths = [2,3,5,7,10]
for i in depths:
    fmdl = RandomForestRegressor(n_estimators=100, max_features='sqrt', max_depth=i)
    fmdl.fit(x_train, y_train)
    print("10-fold CV R2, depth", i, ":", cross_val_score(fmdl, x_train, y_train, cv=10).mean())
```


    10-fold CV R2, depth 2  : 0.00102056654732
    10-fold CV R2, depth 3  : 0.00122405998344
    10-fold CV R2, depth 5  : 0.000980687489795
    10-fold CV R2, depth 7  : -0.00131674140843
    10-fold CV R2, depth 10 : -0.0082639210523


The random forest regression model on latitude and longitude ultimately has an intuitive interpretation. Since restaurants in the dataset in Ontario and Quebec are really just located around Toronto and Montreal, respectively, we can view the coordinates' interaction with a state as an analysis of whether restauraunts in each city get higher ratings the farther west/east/north/south they are located in the city. For random forest, the decision boundary becomes a literal geographic boundary, dividing the city up into sections and assigning score biases to each region.
