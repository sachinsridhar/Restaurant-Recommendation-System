---
title: EDA
notebook: EDA.ipynb
nav_include: 1
---

## Contents
{:.no_toc}
*  
{: toc}


```python
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn.apionly as sns
%matplotlib inline
sns.set_context("notebook")
```




```python
path = "/Users/eesmalling/Data/project121/"
bns = pd.read_csv(path+"business_edit.csv").iloc[:,1:]
usr = pd.read_csv(path+"user_edit.csv").iloc[:,1:]
rvw = pd.read_csv(path+"review_edit.csv").iloc[:,1:]
```


## The Dataset
Yelp's academic dataset contains 5.79 gigabytes of data, consisting of restaurant review text, photos, and data on users and businesses. For the purposes of our project, we used three of the tables available:

+ Business: each row contains information about a single restaurant - name, number of times reviewed, average rating, and location.
+ User: each row contains information about a single user who has posted a review - name, number of reviews, the average star-rating provided by the user, types of compliments given, and number of times the user’s review was marked ‘Useful’.
+ Review: each row corresponds to a single review - date of review, name of the restaurant, user ID and the star rating provided in the review.

### Business
The business data table consists of 51613 restaurants. While Yelpers can write reviews of businesses around the globe, the dataset is limited to several distinct states and regions. Ontario, Canada makes up the largest share of restaurants, followed by Arizona and Nevada. While some of these states are adjacent to each other, the geographical area covered by the dataset is not continuous.



```python
bns.groupby(['state']).size().sort_values().plot.bar()
plt.title('Count of restaurants by state')
sns.despine();
```



![png](EDA_files/EDA_4_0.png)


We also can see below the wide disparity in how many times a restaurant is reviewed. While several popular restaurants were reviewed thousands of times, a large portion of the dataset consists of restaurants only reviewed a couple of times.



```python
plt.hist(bns['review_count'], bins = 50)
plt.yscale('log')
plt.title('Histogram for review counts per restaurant')
sns.despine();
```



![png](EDA_files/EDA_6_0.png)




```python
thresh = np.array([1,3,5,10,25,100,1000], dtype="int")
prop = [np.round(np.mean(bns.review_count<=i)*100,2) for i in thresh]
pd.DataFrame(dict(max_reviews = thresh, percent_total = prop)).transpose()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>max_reviews</th>
      <td>1.0</td>
      <td>3.00</td>
      <td>5.00</td>
      <td>10.00</td>
      <td>25.00</td>
      <td>100.00</td>
      <td>1000.00</td>
    </tr>
    <tr>
      <th>percent_total</th>
      <td>0.0</td>
      <td>7.94</td>
      <td>19.25</td>
      <td>36.05</td>
      <td>59.02</td>
      <td>86.35</td>
      <td>99.69</td>
    </tr>
  </tbody>
</table>
</div>



## User
The user data table consists of 823317 unique users who have reviewed restaurants in the dataset. We see below that, with the exception of a few “power users” that write hundreds of reviews, most users only choose to review restaurants on occasion. For example, 60% of users wrote 10 or fewer reviews:



```python
plt.hist(usr['review_count'], bins = 50)
plt.yscale('log')
plt.title('Histogram for review counts per user')
sns.despine();
```



![png](EDA_files/EDA_9_0.png)




```python
thresh = [1,3,5,10,25,100,1000]
prop = [np.round(np.mean(usr.review_count<=i)*100,2) for i in thresh]
pd.DataFrame(dict(max_reviews = thresh, pct_total = prop)).transpose()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>max_reviews</th>
      <td>1.00</td>
      <td>3.00</td>
      <td>5.00</td>
      <td>10.00</td>
      <td>25.00</td>
      <td>100.00</td>
      <td>1000.00</td>
    </tr>
    <tr>
      <th>pct_total</th>
      <td>12.74</td>
      <td>31.71</td>
      <td>44.01</td>
      <td>60.65</td>
      <td>78.97</td>
      <td>93.69</td>
      <td>99.87</td>
    </tr>
  </tbody>
</table>
</div>



## Review
Noting the discrete regions and low review counts per user and restaurant, we sensed that matrix sparsity would be a big challenge while using matrix factorization to build a review prediction model. In the context of this problem, a sparse matrix would occur when there is little overlap in the restaurants that individual users review, while a “perfectly full” matrix would occur if every user in the data set reviewed every restaurant in the dataset. In the table below, for a given region or combination of regions, we see the number of unique users and restaurants with reviews in that region. The total possible ratings is simply the product of the unique users and restaurants, and coverage is the proportion of these ratings that are actually in the dataset. Coverage, then serves as a measure of matrix sparsity, should we limit our analysis to that region.



```python
cvg = rvw.merge(bns[['business_id','state']], how='left')
cvg_u = cvg.groupby('state')['user_id'].nunique().reset_index()
cvg_b = cvg.groupby('state')['business_id'].nunique().reset_index()
cvg_s = cvg.groupby('state')['stars'].count().reset_index()
cvg_all = pd.DataFrame(cvg_u).merge(pd.DataFrame(cvg_b), how='left').merge(pd.DataFrame(cvg_s), how='left')
cvg_all['mx_size'] = cvg_all.user_id*cvg_all.business_id
cvg_all['mx_coverage'] = round(cvg_all.stars/cvg_all.mx_size,6)
cvg_cols = cvg_all.columns
def state_mx(states, label):
    temp = cvg[cvg.state.isin(states)]
    row = pd.DataFrame(dict(state = label,
                            user_id = temp.user_id.nunique(),
                            business_id = temp.business_id.nunique(),
                            stars = len(temp)), index = np.ones(1))
    row['mx_size'] = row.user_id*row.business_id
    row['mx_coverage'] = row.stars/row.mx_size
    return row
cvg_all = cvg_all.append(state_mx(['IL','WI'], 'IL+WI'))
cvg_all = cvg_all.append(state_mx(['ON','QC'], 'ON+QC'))
cvg_all = cvg_all.append(state_mx(['AZ','NV'], 'AZ+NV'))
cvg_all = cvg_all.append(state_mx(['NC','SC'], 'NC+SC'))
cvg_all = cvg_all.append(state_mx(cvg.state.unique(), 'ALL STATES'))
cvg_all = cvg_all.sort_values('mx_coverage')[cvg_cols].reset_index().drop('index', axis=1)
cvg_all.columns = ['region','unique_users','unique_businesses','total_ratings','possible_ratings','coverage']
```




```python
cvg_all
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
      <th>region</th>
      <th>unique_users</th>
      <th>unique_businesses</th>
      <th>total_ratings</th>
      <th>possible_ratings</th>
      <th>coverage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ALL STATES</td>
      <td>823317</td>
      <td>51613</td>
      <td>2927731</td>
      <td>42493860321</td>
      <td>0.000069</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AZ+NV</td>
      <td>539224</td>
      <td>17102</td>
      <td>1787140</td>
      <td>9221808848</td>
      <td>0.000194</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ON+QC</td>
      <td>113331</td>
      <td>17201</td>
      <td>513411</td>
      <td>1949406531</td>
      <td>0.000263</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AZ</td>
      <td>232460</td>
      <td>10219</td>
      <td>837240</td>
      <td>2375508740</td>
      <td>0.000352</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ON</td>
      <td>85786</td>
      <td>12634</td>
      <td>414444</td>
      <td>1083820324</td>
      <td>0.000382</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NV</td>
      <td>326162</td>
      <td>6883</td>
      <td>949900</td>
      <td>2244973046</td>
      <td>0.000423</td>
    </tr>
    <tr>
      <th>6</th>
      <td>QC</td>
      <td>32684</td>
      <td>4567</td>
      <td>98967</td>
      <td>149267828</td>
      <td>0.000663</td>
    </tr>
    <tr>
      <th>7</th>
      <td>OH</td>
      <td>49062</td>
      <td>4513</td>
      <td>154764</td>
      <td>221416806</td>
      <td>0.000699</td>
    </tr>
    <tr>
      <th>8</th>
      <td>NC+SC</td>
      <td>58213</td>
      <td>3826</td>
      <td>186600</td>
      <td>222722938</td>
      <td>0.000838</td>
    </tr>
    <tr>
      <th>9</th>
      <td>NC</td>
      <td>56923</td>
      <td>3625</td>
      <td>180619</td>
      <td>206345875</td>
      <td>0.000875</td>
    </tr>
    <tr>
      <th>10</th>
      <td>PA</td>
      <td>45665</td>
      <td>3435</td>
      <td>143304</td>
      <td>156859275</td>
      <td>0.000914</td>
    </tr>
    <tr>
      <th>11</th>
      <td>IL+WI</td>
      <td>30047</td>
      <td>2084</td>
      <td>91248</td>
      <td>62617948</td>
      <td>0.001457</td>
    </tr>
    <tr>
      <th>12</th>
      <td>BW</td>
      <td>9599</td>
      <td>1759</td>
      <td>24935</td>
      <td>16884641</td>
      <td>0.001477</td>
    </tr>
    <tr>
      <th>13</th>
      <td>WI</td>
      <td>22116</td>
      <td>1486</td>
      <td>69055</td>
      <td>32864376</td>
      <td>0.002101</td>
    </tr>
    <tr>
      <th>14</th>
      <td>EDH</td>
      <td>6784</td>
      <td>1396</td>
      <td>23751</td>
      <td>9470464</td>
      <td>0.002508</td>
    </tr>
    <tr>
      <th>15</th>
      <td>IL</td>
      <td>8232</td>
      <td>598</td>
      <td>22193</td>
      <td>4922736</td>
      <td>0.004508</td>
    </tr>
    <tr>
      <th>16</th>
      <td>SC</td>
      <td>3450</td>
      <td>201</td>
      <td>5981</td>
      <td>693450</td>
      <td>0.008625</td>
    </tr>
    <tr>
      <th>17</th>
      <td>MLN</td>
      <td>696</td>
      <td>92</td>
      <td>1101</td>
      <td>64032</td>
      <td>0.017195</td>
    </tr>
    <tr>
      <th>18</th>
      <td>HLD</td>
      <td>391</td>
      <td>60</td>
      <td>588</td>
      <td>23460</td>
      <td>0.025064</td>
    </tr>
    <tr>
      <th>19</th>
      <td>FIF</td>
      <td>92</td>
      <td>27</td>
      <td>110</td>
      <td>2484</td>
      <td>0.044283</td>
    </tr>
    <tr>
      <th>20</th>
      <td>C</td>
      <td>118</td>
      <td>23</td>
      <td>168</td>
      <td>2714</td>
      <td>0.061901</td>
    </tr>
    <tr>
      <th>21</th>
      <td>ELN</td>
      <td>91</td>
      <td>19</td>
      <td>117</td>
      <td>1729</td>
      <td>0.067669</td>
    </tr>
    <tr>
      <th>22</th>
      <td>WLN</td>
      <td>60</td>
      <td>18</td>
      <td>87</td>
      <td>1080</td>
      <td>0.080556</td>
    </tr>
    <tr>
      <th>23</th>
      <td>NYK</td>
      <td>89</td>
      <td>12</td>
      <td>101</td>
      <td>1068</td>
      <td>0.094569</td>
    </tr>
    <tr>
      <th>24</th>
      <td>NY</td>
      <td>61</td>
      <td>11</td>
      <td>73</td>
      <td>671</td>
      <td>0.108793</td>
    </tr>
    <tr>
      <th>25</th>
      <td>NI</td>
      <td>50</td>
      <td>8</td>
      <td>58</td>
      <td>400</td>
      <td>0.145000</td>
    </tr>
    <tr>
      <th>26</th>
      <td>01</td>
      <td>15</td>
      <td>6</td>
      <td>24</td>
      <td>90</td>
      <td>0.266667</td>
    </tr>
    <tr>
      <th>27</th>
      <td>ST</td>
      <td>21</td>
      <td>4</td>
      <td>24</td>
      <td>84</td>
      <td>0.285714</td>
    </tr>
    <tr>
      <th>28</th>
      <td>BY</td>
      <td>10</td>
      <td>3</td>
      <td>10</td>
      <td>30</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>29</th>
      <td>ESX</td>
      <td>7</td>
      <td>3</td>
      <td>11</td>
      <td>21</td>
      <td>0.523810</td>
    </tr>
    <tr>
      <th>30</th>
      <td>PKN</td>
      <td>24</td>
      <td>1</td>
      <td>24</td>
      <td>24</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>31</th>
      <td>KHL</td>
      <td>7</td>
      <td>1</td>
      <td>7</td>
      <td>7</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>32</th>
      <td>RCC</td>
      <td>7</td>
      <td>1</td>
      <td>7</td>
      <td>7</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>33</th>
      <td>HH</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>34</th>
      <td>WA</td>
      <td>39</td>
      <td>1</td>
      <td>39</td>
      <td>39</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>35</th>
      <td>WHT</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>36</th>
      <td>CA</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>37</th>
      <td>XGL</td>
      <td>6</td>
      <td>1</td>
      <td>6</td>
      <td>6</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>38</th>
      <td>ZET</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>39</th>
      <td>ABE</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>40</th>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>5</td>
      <td>5</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



We see that there is a tradeoff between the number of reviews and the sparsity of the matrix: the states at the bottom have full coverage but only a few reviews, while the states/regions at the top have many reviews but a relatively sparse matrix. We also note that matrices encompassing two or more states are necessarily more sparse than those for just one state, as it is rare that a user reviews restaurants in multiple states. In order to make sure that the matrix is not too sparse and to consider a large enough set of reviews, we chose to limit our analysis to Ontario and Quebec, Canada's two most populous provinces. The total number of restaurants in these two provinces make up roughly a third of the total number of restaurants in the dataset, while the resulting matrix would be significantly less sparse than using the entire dataset or the larger Arizona and Nevada region.
