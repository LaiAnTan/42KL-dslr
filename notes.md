# dslr-notes

If you havent, it is highly suggested to read [ft_linear_regression](https://github.com/LaiAnTan/42KL-ft_linear_regression) first.

This project goes deeper into the machine learning process, from data cleaning and analysis to classifiers and logistic regression.

The code can be found [here](https://github.com/LaiAnTan/42KL-dslr).

---

## Data Preprocessing

Raw data must be cleaned before it can be analyzed and then used to train a machine learning model.

We remove unnecessary features, as well as seperating the target feature in the dataset.

The dataset should also be split into a training dataset and a testing dataset respectively. This can help to prevent overfitting in the model.

### Missing Data Points

The easiest way to handle missing data points is to disregard the rows containing missing data entirely. However, this is not a good strategy as we lose more data in the process, potentially affecting the accuraccy of the model.

A better way of handling missing data is by **data imputation**. Data imputation is a method for retaining the majority of the dataset's data and information by substituting missing data with a different value.

Methods of data imputation include:

- mean / median / mode imputation: replaces the missing value with mean / median / mode of the feature
- forward / backward fill: carrying forward the last observed non-missing value or by carrying backward the next observed non-missing value
- model imputation: use model to predict missing values

## Data Analysis

Data analysis is a key step in the machine learning process, whereby relationships in the dataset are explored with the goal of improving the quality of the machine learning model.

### Statistical Functions

Statistical functions help to quantify relationships in a dataset.

Common statistical functions include max, min, range, lower quartile, upper quartile, mean, median, mode, interquartile range, etc.

### Sample vs Population

A **population** in statistics refers to **all** the items of interest in a study, which consists of all possible relevant datapoints (even ones that are not collected).

A **sample** in statistics refers to a subset of the population. This is because it is almost impossible to collect all the data present in a paticular population.

Since sample is a a subset of the population, descriptive statistical metrics that are calculated will have to be corrected accordingly, to improve the reliability and accuracy of the metric.

> Further reading: Bessel’s Correction, Degrees of Freedom in Stastistics

#### Concept of Moment

Moments are certain quantitative measures related to the shape of the function's graph.

> We can think of moment as taking the average of distances.

The 1st moment is calculated as follows:

$$
\mu'_1 = \frac{\sum_{i=0}^{n}{x_i}}{n}
$$

where

- $n$ is the number of data points
- $x_i$ represents the $i^{th}$ data point

Intuitively, the formula means that we are taking the distance of each point from zero, and finding the average of those points.

We can observe that this is the just the mean of a dataset.

The 2nd moment is calculated as follows:

$$
\mu'_2 = \frac{\sum\limits_{i=0}^{n}{x_i}^2}{n}
$$

Similarly, the formula means that we are taking the squared distance of each point from zero, and finding the average of those points.

Generally,

$$
\mu'_r = \frac{\sum\limits_{i=0}^{n}{x_i}^r}{n} = E^{\,r}
$$

This formula is known as the **raw moment**, as they are calculated with respect to the origin, 0.

#### Central Moments

However, the raw moment is not very useful when comparing two nth moments of different datasets. To make the distinction clearer, we measure the deviation of the values **without influence from the previous moments**.

For example, if we try to find the 2nd central moment:

$$
\mu_2 = \frac{\sum\limits_{i=0}^{n}({x_i-\mu})^2}{n}
$$

We find that it is the formula for variance.

Notice that we **adjusted for the first moment by subtracting the mean**.

To calculate sample variance, we adjust the formula as follows:

$$
\mu_2 = \frac{\sum\limits_{i=0}^{n}({x_i-\bar{x}})^2}{n - 1}
$$

#### Skewness

The 3rd central moment is known as skewness.

This is the basic formula:

$$
\mu_3 = \frac{\sum\limits_{i=0}^{n}({x_i-\mu})^3}{n}
$$

After adjusting for the variance:

$$
\mu_3 = \frac{1}{n}\frac{\sum\limits_{i=0}^{n}({x_i-\mu})^3}{\sigma^3}
$$

We also have to adjust the formula if we are calculating for sample skewness:

$$
\mu_3 = \frac{n}{(n-1)(n-2)}\frac{\sum\limits_{i=0}^{n}({x_i-\bar{x}})^3}{s^3}
$$

The calculated value is a numerical representation of the skewness in a distribution.

- $\mu_3 = 0$ means that the graph is generally symmetrical
- $\mu_3 \gt 0$ means the tail on the right side of the distribution is longer, i.e. right / positive skew
- $\mu_3 \lt 0$ means the tail on the left side of the distribution is longer, i.e. left / negative skew

#### Kurtosis

The 4th central moment is known as kurtosis.

This is the formula for kurtosis:

$$
\mu_4 = \frac{1}{n}\frac{\sum\limits_{i=0}^{n}({x_i-\bar{x}})^4}{\sigma^4}
$$

Notice again that we do not have to adjust for the previous moment, because the variance and kurtosis are not related. (We can find the kurtosis for a dataset with no variance just fine.)

As usual, adjust the formula if we are calculating for sample kurtosis:

$$
\mu_4 = \frac{n(n + 1)}{(n-1)(n-2)(n-3)}\frac{\sum\limits_{i=0}^{n}({x_i-\bar{x}})^4}{s^4}-\frac{3(n-1)^2}{(n-2)(n-3)}
$$

Kurtosis measures the 'tailedness' of a normal distribution.

Interpreting kurtosis:

- $\mu_4 \approx 3$ means that the distribution of data is **mesokurtic**, i.e. nearly normal
- $\mu_4 \gt 3$ means the distribution of data **leptokurtic**, i.e. long tails
- $\mu_4 \lt 3$ means the tail on the left side of the distribution is longer, i.e.  means the distribution of data **platykurtic**, i.e. short tails

Refer to the graph below:

![kurtosis](/assets/images/kurtosis.png)

### Data Visualisation

Why do we perform data visualisation?

- notice relationships in the dataset.
- make insights and develop an intuition of what the data looks like.
- answer questions related to the dataset that weren't possible with just statistical functions
- detect defects or anomalies

We can use various graphs to display relationships between datapoints, such as:

- histogram
- scatter plot
- pair plot

#### Histogram

![histogram](/assets/images/histogram.png)

The above histograms are to plot the frequency against the magnitude for each feature for each class.

Through these graphs, we can answer the question:

>Which Hogwarts course has a homogeneous score distribution between all four houses?

A **homogenous distribution** means that for every class, the range of values are similarly distributed. For example:

![homogenous](/assets/images/homogenous.png)

In this case, we can see that 'Care of Magical Creatures' has a homogenous score distribution between all four houses. This is because:

- similar histogram shape
- similar range of scores

We want to avoid using homogenous features for classification as they cant classify the dataset well, primarily because the values of the homogenous feature for each class are very similar.

#### Scatter Plot

![scatter_plot](/assets/images/scatter_plot.png)

The above scatter plots aim to visualise the relationship between each pair of features.

> What are the two features that are similar?

**Similar features** have a linear relationship with each other, which means they are highly correlated. For example:

![similar](/assets/images/similar.png)

These two features will have the same effect in classification.

Therefore, the two similar features are 'Arithmacy' and 'Defense against the Dark Arts'.

We also want to avoid using similar features for classification, as both the features will have the same effect in classification, thus creating redundancy.

#### Pair Plot

A pair plot is also known as a scatter plot matrix.

![pair_plot](/assets/images/pair_plot.png)

> What features are you going to use for your logistic regression?

Using the pair plot, and what we have learned about the dataset, we can select the features to be used in the training of the model.

We know that we should not select homogenous and similar features.

We also know that we need features that can classify one class from the rest well.

Since we are only using one feature per logistic regression model, we have to search for the best feature for each class.

After some investigation, I have produced some candidate features for each of the classes in the classifier:

```python
    feature_per_class = {
        "Gryffindor": "Flying", # "History of Magic", "Transfiguration"
        "Hufflepuff": "Ancient Runes",
        "Ravenclaw": "Charms", # "Muggle Studies"
        "Slytherin": "Divination" 
    }
```
The features can be substituted for ones that are commented.

## Feature Selection

The goal of feature selection is to find the best set of features that will produce a machine learning model with the best results.

Through data visualisation, we managed to select a few good features that will be used in our classification model.

> Note that there are various other methods to perform feature selection.

## Classification

Classification is a supervised machine learning method where the model tries to predict the correct label of a given input data.

i.e. given some input data, try to categorise it

**Binary** classification is when the model only classifies data into one of two classes.

**Multiclass** classification is is when the model classfies data into more than two classes.

**Multilabel** classification is when the model assigns multiple labels to a sample, allowing it to belong to more than one category simultaneously.

Algorithms that can be used for classification problems are:

- Logistic Regression
- Support Vector Machine (SVM)
- Perceptron

### Classifiation Models (Strategies)

#### One vs All / One vs Rest (OvR)

The One vs Rest strategy uses multiple binary classifiers together for multiclass classification.

For a dataset of $N$ classes, $N$ binary classification models are created.

- one model for each class to predict (in this case logistic regression)
- each model is used to predict if the sample belongs to a specific class, for example
  - model $A$ vs not $A$ is used to predict if a sample belongs to the class $A$.

Run the sample through all $N$ models, and take the one with the highest value as the final predicted class. (max)

#### One vs One (OvO)

The One vs One strategy is another method that uses multiple binary classifiers together for multiclass classification.

Compared to OvR classification, this strategy seperates the dataset into one dataset for each class versus every other class.

For a dataset of $N$ classes, $\frac{N \times (N-1)}{2}$ binary classification models are created.

For example, consider a multiclass classification problem with 4 classes. We would divide it into six binary classification problems:

- $C_1$ vs $C_2$
- $C_1$ vs $C_3$
- $C_1$ vs $C_4$
- $C_2$ vs $C_3$
- $C_2$ vs $C_4$
- $C_3$ vs $C_4$

The models will perform a 'vote', where each model will submit a vote for their predicted class. The class with the most number of votes will be the predicted class.

In case of a tie, tiebreaking strategies are required, such as:

- using confidence scores from models
- random selection between tied classes
- predetermined class priority

#### Softmax

Softmax function is a generalization of logistic regression that inherently handles multiclass classification.

In softmax, each of the classes is given a proper probability that a sample belongs to it.

The softmax function is as follows:

$$
P(y=j|x,\{w_k\}_{k=1...K}) = \frac{e^{x^\top w_j}}{\sum\limits_{k=1}^K e^{x^\top w_k}}
$$

where

- $P(y=j|x,{w_k}_{k=1...K})$ is the conditional probability that the output y belongs to class j given input $x$ and the set of weight vectors $w_k$ for all $K$ classes
- $x$ is the input feature vector
- $w_k$: is the weight vector for class $k$. There are $K$ such vectors, one for each class
- $x^⊤ w_j$ is the dot product of $x$ and $w_j$, represents the score (or logit) for class $j$
- $e^{(x^⊤ w_j)}$ is the exponential of the score for class $j$
- $\sum\limits_{k=1}^K e^{(x^⊤ w_k)}$ is the sum of exponentials of scores for all K classes. It acts as a normalizing factor to ensure probabilities sum to 1.
- $K$ is the total number of classes

Pretty complicated, will explore this in a future project.

## Logistic Regression

Logistic Regression is a model used for binary classification. This means that the output prediction of a logistic regression model can only be either 0 (`False`) or 1 (`True`).

A linear equation does not work for this purpose as it is **unbounded**.

![image](/assets/images/linear.png)

However, the sigmoid curve fits the criteria of mapping independent variable $x$ where $x \in [-\infty, +\infty]$ to dependent variable $y$ where $y \in [0, 1]$.

![image](/assets/images/sigmoid.png)

The function of a sigmoid curve is as follows:

$$\sigma(x)=\frac{1}{1+e^{-x}}$$

Fortunately, we can easily fit a linear equation ${\hat{y}=mx+c}$ to a sigmoid curve by applying the sigmoid function.

### Hypothesis

To obtain a prediction, the following formula is used:

$$\hat{y}=\sigma(mx + c)$$

> Note: The hypothesis equation for Logistic Regression with multiple features is: $$\hat{y}=\sigma(w\cdot x+b)$$
> where
>
> - $\hat{y}$ is the predicted value for a sample
> - ${w}$ is a ${n\times1}$ matrix of weights where ${n}$ is the number of features (i.e. gradients)
> - ${x}$ is a ${n\times1}$ matrix of data points in a sample where ${n}$ is the number of features
> - ${b}$ is the bias (i.e. intercept)
> - $\sigma(x)$ is the sigmoid function
>
> which can be rewritten as $\hat{y}=\sigma{(w_0x_0+w_1x_1+\cdots+w_nx_n +b)}$, which is just the linear equation but with multiple features instead of just one.
>
> However, *with many features comes great responsibility*. (Pun intended)
> The sigmoid curve obtained might look different / produce unintended results which depend on multiple factors of the features used, such as their correlation with each other.

### Cost Function

The cost function of Logistic Regression is as follows:

$$
C=-\frac{1}{m}\sum\limits_{i=0}^{m}[\,y\log(\hat{y})+(1-y)\log(1-\hat{y})\,]
$$

where

- $m$ is the number of data samples
- $\hat{y}$ is the predicted value for a sample
- and ${y}$ is the actual value of that sample

To explain the cost function, we will describe two cases:

When ${y = 0}$, i.e. the sample is not a member of the class, the cost for one sample is:

$$C=-1[\,0+(1-0)\log(1-\hat{y})\,]$$
$$C=-1\log(1-\hat{y})$$

When ${y = 1}$, i.e. the sample is a member of the class, the cost for one sample is:

$$C=-1[\,1\log(\hat{y})+(1-1)\log(1-\hat{y})\,]$$
$$C=-1\log(\hat{y})$$

Referring to the image below:

![image](/assets/images/cost.png)

The red graph represents $C=-1\log(\hat{y})$, and the blue graph represents $C=-1\log(1-\hat{y})$.

In both graphs, we can see that as the prediction gets closer to the actual value, the cost function decreases, and vice versa. That is exactly how a cost function should behave.

Now, we just have to take the average over all samples to calculate the cost for one epoch.

### Differentiating the Cost Function

Now we will differentiate the cost function to obtain the derivatives w.r.t. gradient and intercept, so that we can use them in gradient descent.

We start with three equations:

The cost of one sample, $L$ :
$$
L=-[y\log a+(1-y)\log(1-a)]
$$

The hypothesis function :
$$
a=\sigma(z)=\frac{1}{1+e^{-z}}
$$

The equation for a straight line:
$$
z(x)=mx+c
$$

Calculations in the image below:

![image](/assets/images/dcost.png)

> I am so not writing latex for this...

after the calculations above, we end up with:

$$
\frac{\delta L}{\delta m}=(\hat y-y)x
$$
$$
\frac{\delta L}{\delta c}=\hat y-y
$$

which we can use in gradient descent.

## Gradient Descent

1. Stochastic Gradient Descent

    This variant of gradient descent updates the parameters after each sample.

    In a dataset with $S$ samples, the parameters are updated $S$ times per epoch.

    Stochastic gradient descent is used for large datasets to make convergence to the minimum faster.

2. Batch Gradient Descent

    This variant of gradient descent updates the parameters after the whole dataset.

    Batch gradient descent is more resistant against outliers.

3. Mini Batch Gradient Descent

    This variant of gradient descent seperates the dataset into batches. The parameters are updated after each batch is processed.

    Mini Batch gradient descent aims to capture the advantages of both Stochastic and Batch gradient descent.

    > Note: Batch size is usually a power of 2.

## Outcome

I decided to fit a logistic regression model for each of the 13 features for every class, totalling to 52 models, as shown in the graph below:

![logreg](/assets/images/logreg.png)

In the graph, we can observe that there are sigmoids which are more pronounced, as well as sigmoids which are not very obvious.

The sigmoids that are not very obvious display what happens when we select a feature that is bad at classifying the specific class. Minimizing the cost function usually plataus at around the range of 0.4 - 0.8.

The sigmoids that show an obvious distinction between 0 and 1 are the models that are able to classify the specific class well. In this case, the minimum of the cost function is below 0.1.

If we use the feature with the lowest cost from each of the classes, we get a classifier with a precision accuracy of 98.25%, out of 400 samples for the test dataset!

## Sources

Data Imputation
https://medium.com/@pingsubhak/handling-missing-values-in-dataset-7-methods-that-you-need-to-know-5067d4e32b62
https://www.analyticsvidhya.com/blog/2021/10/handling-missing-value/

Moments

https://youtu.be/ISaVvSO_3Sg?si=ccrfHNuIZX535K2a
https://en.wikipedia.org/wiki/Central_moment

OvO, OvR

https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/

Logistic Regression:

https://youtube.com/playlist?list=PLuhqtP7jdD8Chy7QIo5U0zzKP8-emLdny&si=cyhdGOXxUotPwjNP

https://youtu.be/smLdMzVlmyU?si=4yJYKLyDzx1_8YmV
https://youtu.be/5y35Rll3yIE?si=L4VE3k36u6Ty6XN3
https://youtu.be/xgY05vLWicA?si=HczOnBP_sb8g08jy

Mini Batch GD:

https://datascience.stackexchange.com/questions/18414/are-there-any-rules-for-choosing-the-size-of-a-mini-batch
