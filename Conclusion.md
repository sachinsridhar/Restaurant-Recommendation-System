---
title: Conclusion
notebook: Modeling.ipynb
nav_include: 4
---

## Contents
{:.no_toc}
*  
{: toc}


## Results, Conclusions and Further Work

Below we see the performance of the ensemble model on the holdout (test) dataset, as well as the training dataset.

    training r^2 for regression model:                                       0.269712582918
    training r^2 for regression + matrix model:                              0.269712582918
    training r^2 for regression + matrix + Random Forest regression model:   0.27920343767


    holdout r^2 for regression model:                                       0.0928745193673
    holdout r^2 for regression + matrix model:                              0.0928745193673
    holdout r^2 for regression + matrix + Random Forest regression model:   0.0972615487885


As we noted earlier, the predicted residuals from the matrix factorization step do not improve the accuracy of the model, because they are practically zero.
