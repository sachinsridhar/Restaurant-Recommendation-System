---
title: Literature Review
nav_include: 7
---

## Yelp Dataset Challenge

The Yelp Dataset Challenge is a recurring challenge that is put forth every season to encourage participants to churn out valuable insights based on the data shared by Yelp. Among several challenges, such as Sentiment Analysis, Photo Classification and Graph Mining, lies an overall objective of building a Recommender System.


## Bibliography

1. Becker N. ["Matrix Factorization for Movie Recommendations in Python"](https://beckernick.github.io/matrix-factorization-recommender/) *Personal website*. Github, 10 Nov. 2016.
2. Nee D. ["Collaborative Filtering Using Alternating Least Squares"](http://danielnee.com/2016/09/collaborative-filtering-using-alternating-least-squares/) *Personal website*. Github, 17 Sept. 2016.
3. Koren Y. and Bell R. ["Advances in Collaborative Filtering"](https://datajobs.com/data-science-repo/Collaborative-Filtering-%5BKoren-and-Bell%5D.pdf) *Recommender Systems Handbook.* Ed. Ricci F., Rokach L., Shapira B., Kantor P. Boston: Springer, 2011. 145-187. Print.
4. Koren Y., Bell R. and Volinsky C. ["Matrix Factorization Techniques for Recommender Systems"](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf) *Computer* 42.8 (2009): 42-49. Print.
5. TBD code for matrix factorization


## Literature Review

Many analyses have been performed for the Netflix dataset challenge - which was analogous to our problem - building a recommender system of movies, based on past movie preferences of users. We reviewed reference 1 to understand what has been done in the past to solve this problem.

In order to gain some insight into the implementation of Matrix Factorization to build recommender systems we reviewed reference 2.

Collaborative Filtering is commonly used in recommender systems. The idea is that if we have a large set of item-user preferences, we use collaborative filtering techniques to predict missing item-user preferences. For example, we have the purchase history of all users on an eCommerce website. You use collaborative filtering to recommend which products a user might purchase next. The key assumption here is people that agreed in the past (purchased the same products) will agree in the future.
 
A major advantage of collaborative filtering is that it is not dependent on the domain, and yet, it can address data aspects that are often elusive and difficult to profile using content filtering. While generally more accurate than content-based techniques, there is one major limitation with collaborative filtering. We encounter the cold-start problem i.e. the system is unable to address the system’s new products and users. 
 
The two primary areas of collaborative filtering are the neighborhood methods and latent factor models:

Neighborhood methods are centred on computing the relationships between items or, alternatively, between users. The item-oriented approach evaluates a user’s preference for an item based on ratings of “neighboring” items by the same user. A product’s neighbors are other products that tend to get similar ratings when rated by the same user. 
 
Latent factor models are an alternative approach that tries to explain the ratings by characterizing both items and users on factors inferred from the ratings patterns. In a sense, such factors comprise a computerized alternative to manually building profiles of users and items.


