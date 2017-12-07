---
title: Conclusion
notebook: Modeling.ipynb
nav_include: 4
---

## Results

Below we see the performance of the ensemble model on the holdout (test) dataset, as well as the training dataset.

    training r^2 for regression model:                                       0.269712582918
    training r^2 for regression + matrix model:                              0.269712582918
    training r^2 for regression + matrix + Random Forest regression model:   0.27920343767


    holdout r^2 for regression model:                                       0.0928745193673
    holdout r^2 for regression + matrix model:                              0.0928745193673
    holdout r^2 for regression + matrix + Random Forest regression model:   0.0972615487885


## Conclusions and Further Work

Several things stand out from the results displayed above. In examining the training scores, we see that a bulk of the ensemble's explanatory power comes from the baseline regression. As we noted before, the predictions of the residuals from the matrix model were practically zero, explaining why the score did not increase with the addition of this step. The random forest regression model marginally improves the score, though its contribution is minimal when compared to the baseline regression.

We also notice that the out of sample score is significantly lower than the training score, indicating that the model does not generalize well. This makes sense, in part because the baseline regression and matrix factorization were performed on indvidual user and restaurants. It is encouraging, however to see that the random forest model also improves the accuracy out of sample.

While it is somewhat disappointing to see that the matrix factorization model did not contribute meaningfully to the final results, as it was the focus of a majority of our energy during this project, it also illustrates well how challenging it is to implement. We had to carefully filter and shape our data so the technique would work at all, resulting in countless iterations of running code. We then had to tune several hyperparameters, a computationally and time-intensive task. Our initial choice of lambdas was too small, resulting in an overfit matrix and a negative test $R^2$. After trying more values, it became clear that the optimal lambda was the one that reduced the matrix to zero. 

Because of our struggles and successes with the matrix factorization technique, we now have a sense for how it could be better performed in the future with more time and resources...


