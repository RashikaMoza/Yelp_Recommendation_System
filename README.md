
                   
 YELP SENTIMENT ANALYSIS AND RECOMMENDATION SYSTEM




INTRODUCTION
Yelp is one channel, but many restaurants document verbal feedback, gather feedback from questionnaires, or even from other review sites. When faced with a huge amount of text, restaurants need a way to objectively interpret their reviews. By building a way to relate text-based feedback to a star rating, we can help these restaurants understand how they would rate on Yelp. 

BACKGROUND 

We are predicting Yelp user ratings by implementing Natural Language Processing on user reviews. By examining the user review text with various techniques involved in text mining and text analysis. We have trained the model to predict ratings using algorithms like: Naïve Bayes Multinomial Algorithm, Support Vector Machine (SVM), Random Forest and evaluated the performance based on confusion matrix and classification report consisting of precision, recall and f1-measure. 
Further, we have predicted the user ratings for food related restaurants using various models like Baseline Only, KNN Baseline (user-based and item-based collaborative filtering method) and matrix factorization methods like Singular-Value Decomposition (SVD) and SVDpp. 
Finally, we have evaluated the performance by calculating the average RMSE and MAE on 3-fold cross validation and implementing review-based restaurant recommendation to recommend top-N restaurants to the user.

OBJECTIVE 

In this project, we will be applying various machine learning algorithms to predict text-based feedback to star rating. In short, we are precisely trying to analyze on the following points:
1.	Performing sentimental analysis on the input given by the user 
2.	Analyzing and recommending users based on their preferences/ choices
3.	Predicting the polarity of the review (positive/negative) using supervised methods
4.	Understand which algorithm has the highest accuracy score for the given data set
5.	Analyzing the speed performance of the algorithms and confusion matrix

METHODOLOGY
In this project we have used various classification algorithms to predict star rating based on text feedback such as: 
1.	Naive Bayes Multinomial: It is used for discrete counts. For example, let’s say, we have a text classification problem. We have count how often word occurs in the document, we can think of it as “number of times outcome number x is observed over the n trials

2.	SVM: It is a non-probabilistic binary linear classifier. In this study, SVM Model represents each review in vectorized form as a data point in the space. This method is used to analyze the complete vectorized data and the key idea behind the training of model is to find a hyperplane


3.	Random Forest: Like its name implies, consists of large number of individual decision trees that operate as an ensemble. Each individual tree in the random forest spits out a class prediction and the class with the most votes becomes our model’s prediction

Surprise (Simple Python Recommendation System Engine) is a recommendation system library, which is one of the scikit series. Simple and easy to use, while supporting a variety of recommendation algorithms (basic algorithm, collaborative filtering, matrix decomposition, etc.)
We predicted the user ratings for food related restaurants using various models like:
1.	Baseline Only
2.	KNN Baseline (user-based and item-based collaborative filtering method)

SPECIFICATIONS

The data has been fetched from https://www.kaggle.com/yelp-dataset/yelp-dataset
•	Number of user reviews: 5,200,000
•	Number of businesses: 174,000
•	Data spans 11 metropolitan areas

Table names and Descriptions:
1.	Business:   Consist of columns like business id, name, address, city, state 
2.	Reviews:    Consist of columns like of user id, business id, review id, stars, text, useful
3.	User: 	      Consist of columns like name, user id, review count, useful
4.	Tips:           Consist of columns like user id, business id, text, compliment count  


SUMMARY AND OBSERVATION OF EACH ALGORITHM 

Naive Bayes Multinomial
We have used multinomial Naive Bayes classification, which assumes that P(ri |s) is a multinomial distribution for all i. This is a typical choice for document classification, because it works well for data that can be turned into counts, for example weighted word frequencies in the text.
                       

 

SVM

It is a non-probabilistic binary linear classifier. In this study, SVM Model represents each review in vectorized form as a data point in the space. This method is used to analyze the complete vectorized data and the key idea behind the training of model is to find a hyperplane.
These are enhanced versions of perceptron, in that they eliminate the non-uniqueness of solutions by optimizing the margin around the decision boundary and handle non-separable data by allowing misclassifications. A parameter C controls overfitting. When C is small, the algorithm focuses on maximizing the margin, even if this means more misclassifications, and for large values of C, the margin is decreased if this helps to classify more examples correctly. In our experiments, we use linear SVMs for multiclass classification. The tolerance of the convergence criterion is set to 0.001. For each feature extraction method, we do internal 3-fold cross validation to choose the value of C that gives the highest accuracy. It turns out that C = 1.0 works best every time.
   

 
RANDOM FOREST 

Random Forest is a collection of K classifiers h1(x), h2(x), hK(x). Each of these classifiers votes for one class and every instance is classified base on the majority class. Every instance of the n training set instances is drawn at random and some instances are nor used in building each tree. Those instances are useful in the internal estimation of the length and correlation of the forest. Random forests are computationally effective and offer good prediction performance. 
 
 


RECOMMENDATION  
We combine item-based and user-based collaborative filtering to predict ratings of a user for a given restaurant. Item-based collaborative filtering will consider the restaurants the user has rated and predict the rating according to scores of these similar restaurants. 
We use 𝑟𝑎𝑡𝑒𝑑(𝑥) to denote the set of restaurants that user 𝑥 has rated, and in order to reduce time complexity, we only select 20 restaurants by random sampling when calculating the similarity value. 
User-based collaborative filtering considers the similar users’ ratings, where 𝑁(𝑥) is the set of similar users we get by k-NN algorithm. 𝑠𝑖n𝑥, 𝑏) is the similarity value we get by transforming the Euclidian distances between users into the range of (0, 1).
Linear regression is a machine learning approach to build the linear relationship between dependent variables and several dimensions of independent variables. Our experiments show that when we set w1, w2, w3 and bias to be 1.0, 0.3, 0.1 and -1.5, it will produce satisfying results. Finally, if the prediction score value of user x for restaurant y is no less than 4.0, then we will recommend this restaurant to the user, otherwise we will not recommend it. 

CONCLUSION
We have presented a feature selection followed by a classification-based approach to the recommendation of ‘useful’, ‘funny’ and ‘cool’ reviews in Yelp site. We have considered various feature selection techniques and examined their performance in terms of accuracy and the number of features included in the classification approach. The learning task proved that Random Forest was more robust to the presence of noisy features, while Naive Bayes achieved best accuracy when only considering top ranked features. User features, such as the user average helpfulness votes and the percentage of useful/funny and cool reviews the user writes, and structural features, such as the number of words/complex words and sentences, proved to be most useful in terms of classification performance. Business features were less successful. Such results give us an insight of what makes reviews ‘useful’, ‘funny’ or ‘cool’ in Yelp.com.
 
Accuracy Score: 
•	Naïve Bayes: 0.60966
•	SVM Classifier: 0.460433
•	Random Forest Classifier: 0.412466


