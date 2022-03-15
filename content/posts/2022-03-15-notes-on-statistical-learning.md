---
title: "Notes on Statistical Learning"
date: 2022-03-15T12:18:00-05:00
tags: ["stats"]
author: "Rui Qiu"
showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: true
description: "Notes of ISLR2."
# canonicalURL: "https://canonical.url/to/page"
disableHLJS: true # to disable highlightjs
disableShare: true
hideSummary: true
searchHidden: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
math: true
cover:
    image: "<image cover>" # image path/url
    alt: "<alt text>" # alt text
    caption: "<text>" # display caption under cover
    relative: false # when using page bundles set this to true
    hidden: true # only hide on current single page
editPost:
    URL: "https://github.com/rexarski/blog/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

Source: An Introduction to Statistical Learning (ISLR) 2nd edition

Chapter 1 & 2: Readings
- [x] What Is Statistical Learning? (p.1-28)
- [x] Assessing Model Accuracy (p.29-51)

Chapter 3: Readings
- [x] Simple and Multiple Linear Regression (p.59-82)
- [x] Assumptions and Other Potential Problems (p.82-119)

Chapter 5: Readings
- [x] Resampling Methods (p.175-190)

Chapter 6: Readings
- [x] Best Subset and Stepwise Selection (p.205-210)
- [x] Estimating Error w/ Cross-Validation (p.210-214)
- [x] Ridge Regression and the Lasso (p.214-228)
- [x] PCR and PLS (p.228-244)

Chapter 7: Readings
- [x] Polynomial Regression and Step Functions (p.265-270)
- [x] Splines and GAMs (p. 271-287)

Chapter 4: Readings
- [x] Logistic Regression (p.130-138)
- [x] Intro. to Classification
- [x] KNN (p.37-42, p.104-109,p.129-130)
- [x] Discriminant Analysis (p.138-150)
- [x] Classification Wrap-up (p.151-154)

Chapter 8: Readings
- [ ] Decision and Classification Trees (p.303-316)
- [ ] Bagging, Random Forests, and Boosting (p.316-324)

Chapter 10: Readings
- [ ] Introduction to Neural Networks
- [ ] Unsupervised Learning and Principal
- [ ] Components Analysis (p.373-385)
- [ ] K-Means and Hierarchical Clustering (p.385-401)

***

# Notes to ISLR
## Chapter 2 Statistical Learning

- input variables: **predictors**, independent variables, features, or sometimes just variables.
- output variables: **response**, dependent variables.

$$Y=f(X)+\epsilon$$
- The random error term $\epsilon$ is independent of $X$ and has mean zero, $f$ represents the systematic information that $X$ says about $Y$.
- $f$ -> prediction and inference.
	- $\hat{Y} =\hat{f}(X)$, resulting prediction of $Y$
- The accuracy of $\hat{y}$ depends on two quantities, *reducible error* and *irreducible error*.
	- Reducible error can be reduced by improving the accuracy with an appropriate statistical learning method.
	- Variability associated with $\epsilon$ is the irreducible error, which may contain unmeasurable variation.

$$E(Y-\hat{Y})^2=E[f(X)+\epsilon-\hat{f}(X)]^2=[f(X)-\hat{f}(X)]^2+Var(\epsilon)$$
- Prediction -> $\hat{f}$ is often treated like a black box.
- Inference -> $\hat{f}$ need to know the explicit form.
- Pick the right tool depending on what the goal is.
- Linear -> good for inference but not good for prediction.
- To measure the performance of $\hat{f}$ :
	- parametric: set the linear form/shape -> use (ordinary) least squares, for example.
	- non-parametric:
		- do not make explicit assumptions about the function $f$, for example, a *thin-plate spline*.
- Trade-off between prediction accuracy and model interpretability

> Note:
> **Lasso**: relies the linear model but uses an alternative fitting procedure for coefficient estimation, which are more restrictive (by setting some of the coefficients zero), this makes lasso a less flexible approach than linear model. But it is more interpretable since some predictors are "removed."
> **Generalized additive models (GAMs)**: linear -> non-linear models -> more flexible but less interpretable.

- **Supervised** vs **non-supervised** learning:
	- with or without a response
- **Semi-supervised** example: A set of $n$ observations, for $m<n$ observations we have both predictors and responses, but the remaining $n-m$ observations do not.

## Assessing Model Accuracy

- **Mean squared error (MSE)**:

$$MSE=\frac1n\sum^n_{i=1}(y_i-\hat{f}(x_i))^2$$

- Pick the method that minimizes MSE on testing data, or when without testing, simply pick the one generating the minimal MSE on training data.
- **The Bias-Variance Trade-Off**

$$E(y_0-\hat{f}(x_0))^2=Var(\hat{f}(x_0))+[Bias(\hat{f}(x_0))]^2+Var(\epsilon)$$

- This is the expected test MSE at $x_0$. Good fit -> minimize both variance and bias.
- **Variance**: the amount by which $\hat{f}$ would change if we estimated it using a different training data set. Different data -> different $\hat{f}$.
	- More flexible statistical methods -> higher variance.
- **Bias**: the error that is introduced by approximating a real-life problem.
	- More flexible statistical methods -> lower bias.

For classification settings,
- **Bayes classifier**
	- Error rate: $\frac1n\sum^n_{i=1}I(y_i\not=\hat{y}_i)$
		- of course, with training and testing data.
	- Bayes classifier
		- Bayes classifier -> lowest possible test error rate -> **Bayes error rate**
		- The overall Bayes error rate:

$$1-E\left(\max_j\Pr(Y=j|X)\right)$$
- **K-Nearest Neighbors**
	- get the conditional probability for class $j$ based on $K$ points in the training data that are closest to $x_0$, as $\mathcal{N}_0$

$$\Pr(Y=j|X=x_0)=\frac1K\sum_{i\in\mathcal{N}_0}I(y_i=j)$$

## Chapter 3 Linear Regression

Also lecture on 2022-01-24.

- In marketing, there's a term called **synergy effect**, which is an **interaction effect** in statistics.

### Simple Linear Regression

- *Intercept*, *slope*. *Coefficients.*

$$y = \beta_0 + \beta_1 X + \epsilon$$
$$\hat{y} = \hat{\beta_0}+\hat{\beta_1}X$$
- Most common approach to measure *closeness* is the **least squares** criterion.
	- **residual** $e_i = y_i -\hat{y}_i$
	- **residual sum of squares (RSS)** $\text{RSS}=e_1^2+e_2^2 +\cdots + e_n^2$.
	- Using calculus, calculate $\hat{\beta}_1$ and $\hat{\beta}_0$.
- Assessing the accuracy of the coefficient estimates:
	- True regression line (population regression line) vs the *least squares regression line*
	- On average, the mean of coefficient estimates are the same with sample mean. **Unbiased.**
	- However, the variance of coefficient estimate, which is associated with **standard error of $\hat{\mu}$** is:
$$Var(\hat{\mu})=SE(\hat{\mu})^2=\sigma^2/n$$
where $\sigma$ is the standard deviation of each of the realization $y_i$ of $Y$.
- Roughly, the **standard error** -> the average amount $\hat{\mu}$ deviates from the true value $\mu$.
- $$SE(\hat{\beta}_0)^2 = \sigma^2\left[\frac1n + \frac{\bar{x}^2}{\sum^n_{i=1}(x_i-\bar{x})^2}\right]$$
$$SE(\hat{\beta}_1)^2=\frac{\sigma^2}{\sum^n_{i=1}(x_i-\bar{x})^2}$$ where $\sigma^2=Var(\epsilon)$

- In general, $\sigma^2$ is unknown, but can be estimated from data as **the residual standard error**

$$RSE=\sqrt{RSS/(n-2)}$$

- Standard errors -> **confidence intervals**. A 95% confidence interval -> a range of values such that with 95% probability, the range will contain the true unknown value of the parameter:

$$\hat{\beta}_1\pm 2\cdot SE(\hat{\beta}_1)$$

- **Hypothesis testing**
	- **t-statistic** is $t = \frac{\hat{\beta_1}-0}{SE(\hat{\beta}_1)}$ measures **the number of standard deviations that** $\hat{\beta}_1$ is away from 0.
	- The probability of **observing any number equal to $|t|$ or larger in absolute value assuming $\beta_1=0$** is called **the p-value**.
	- Small p-value: unlikely to observe such a substantial association between the predictor and the response due to chance, in the absence of any real association between the predictor and the response -> Therefore, there is an association between the predictor and the response -> **reject the null hypothesis**

- Assessing the accuracy of the model
	- with typically two related quantities:
		- the **residual standard error (RSE)**
		-  and the **$R^2$ statistic**
		- (Hey, and **F-statistic**)
- RSE -> large -> lack of fit -> doesn't fit well.
- $R^2$
	- the proportion of variance explained:

$$R^2=\frac{TSS-RSS}{TSS}=1-\frac{RSS}{TSS}$$

where $TSS=\sum(y_i-\bar{y})^2$ is the **total sum of squares**

- Adjusted-$R^2 = 1 - \frac{RSS/(n-d-1)}{TSS/(n-1)}$

- Correlation

$$Cor(X,Y)=\frac{\sum^n_{i=1}(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum^n_{i=1}(x_i-\bar{x})^2}\sqrt{\sum^n_{i=1}(y_i-\bar{y})^2}}$$

In simple linear regression setting, $R^2=r^2$.

### Multiple Linear Regression
Need RSS too

$$RSS=\sum^n_{i=1}(y_i-\hat{\beta}_0-\hat{\beta}_1x_{i1}-\cdots -\hat{\beta}_px_{ip})^2$$

- Questions to ask?
	- 1. Is there a relationship between the response and predictors?
		- Hypothesis test on at least one of the predictor coefficient is non-zero with **F-statistic**
		- $$F=\frac{(TSS-RSS)/p}{RSS/(n-p-1)}$$
		- **When there is no relationship between the response and predictors, F-stats is close to 1.** Otherwise, F-stats greater than 1.
		- Also can F-test on partial effects with a subset of predictors. where $TSS$ is replaced with $RSS_0$ and the numerator degree $p$ is replaced with the number of coefficients in the subset as $q$.
	- 2. Do all the predictors help to explain $Y$?
		- "variation selection" which will be talked about more in [[#Chapter 6 Linear Model Selection and Regularization]]
		- Metrics for judging a model:
			- **Mallow's $C_p$**
			- **Akaike information criterion (AIC)**
			- **Bayesian information criterion (BIC)**
			- and **adjusted $R^2$**
		- Selection method:
			- **Forward selection**: begin with null model -> more complex
			- **Backward selection**: begin with full model -> simpler
			- **Mixed selection**
	- 3. How well does the model fit?
		- Caution for **interaction effect (or synergy)** when specific combination of predictors -> overestimate/underestimate the response.
	- 4. Given a set of predictor values, what response value should we predict, and how accurate is the prediction?
		- The **prediction interval** which is much wider than the **confidence interval**.


### Other Considerations in the Regression Model
- Two of the most important assumption of the relationship between response and predictors are **additive** and **linear**.
- **Additive assumption**: the association between a predictor $X_j$ and the response $Y$ does not depend on the values of the other predictors.
- What if we remove the **additive assumption**?
	- With the introduction of an interaction term, this is not any more.
	- If the p-value of an interaction term is significant -> the model is not additive.
- The **hierarchical principle** is: if we include an interaction in a model, we should also include the main effects, even if the p-values associated with their coefficients are not significant.

- Polynomial regression.

- Potential problems:
	- Non-linearity. -> Residual vs fitted plot -> use transformation to fix
	- Correlation of error terms. -> Heteroscedasticity -> Residual vs fitted plot
	- Non-constant variance of error terms.-> studentized residual vs fitted plot to see (weighted least squares)
	- Outliers. -> studentized residual vs leverage -> high residual
	- High-leverage points.
		- leverage statistic.
		- between $1/n$ and 1, average leverage $(p+1)/n$.
		- If exceeding $(p+1)/n$ then it's high-leverage.
	- Collinearity.
		- High correlation between two predictors -> multicollinearity -> **variance inflation factor (VIF)**
		- $$VIF(\hat{\beta}_j)=\frac1{1-R^2_{X_j|X_{j-1}}}$$

## Chapter 5 Resampling Methods

Lecture on 2022-01-31

> Repeated draw different samples from the training data -> resampling.

- **cross-validation** and **bootstrap**
- model assessment -> model selection
	- cv -> used to estimate the test error associated with a given statistical learning method
	- bootstrap -> used to measure a parameter estimate or of a given statistical learning method

### Cross-Validation

- **training set**, **hold-out or validation set.** (Depending on the mechanism, you probably want or not to slice a part of data as the testing data beforehand.)
- validation set -> a typical error rate like **MSE**

- The $k$-fold CV estimate is by averaging $$CV_{(k)}=\frac1k\sum^k_{i=1}MSE_i$$

- That's **k-fold cv**

- What about **Leave-One-Out CV (LOOCV)** -> eventually, the LOOCV estimate for the test MSE is the average of these $n$ test error estimates: $$CV_{(n)}=\frac1n\sum^n_{i=1}MSE_i$$
 - **special shortcut to make the cost of LOOCV the same as that of single model fit** $$CV_{(n)}=\frac1n \sum^n_{i=1}\left(\frac{y_i-\hat{y}_i}{1-h_i}\right)^2$$
 - where $h_i$ is the leverage between 1 and 1/n


- major advantage of **LOOCV**:
	- it has **far less bias**
	- performing LOOCV always yields the same results -> **no randomness in the training/validation set splits.**
- disadvantage
	- computationally expensive
	- identical training sets -> model estimates have high correlation -> in danger of overfitting

-> k-fold CV often gives more accurate estimates of the test error rate than does LOOCV due to **bias-variance trade-off.**

- From the bias reduction perspective, LOOCV > k-fold
- variance: LOOCV has higher variance than does k-fold CV with $k<n$

### The Bootstrap

- The **bootstrap** is a widely applicable and extremely powerful statistical tool that can be used to quantify the uncertainty associated with a given estimator or statistical learning method.
- The bootstrap approach allows us to use a computer to emulate the process of obtaining new sample sets -> estimate the variability of an estimated parameter without generating additional samples. ==Rather than repeatedly obtaining independent data sets from the population, we instead obtain distinct data sets by repeatedly sampling observations from the original data set.==


## Chapter 6 Linear Model Selection and Regularization

Lecture on 2022-02-07.

- Subset selection
	- best subset selection
	- stepwise selection: forward selection, backward selection
	- choosing the the optimal
	- AIC, BIC, Cp and $R^2$ (metrics)

- Solution to ($n \approx p$)
	- Subset selection (this week)
	- Shrinkage (next week)
	- Dimension Reduction
		- projecting $p$ predictors to $M<p$ dimensional subspace.

### Best subset selection
- simple, but suffers from computational limitations
- the number of possible models grows exponentially
- enumerating to find the model with the best metric.

> Forward/backward selection saves time/computation but will be rather "linear" in finding the suitable subset of the predictors.

Estimating test error: two approaches:
- indirectly estimate test error by **making an adjustment**
-

Measures of comparison
- AIC and Mallow's Cp are equivalent in terms of linear regressions
- BIC
- Adjusted-R^2

$$Adjusted-R^2=1-\frac{RSS/(n-d-1)}{TSS/(n-1)}$$

$d$ is the number of variables, TSS is the total sum of squares. The newly added flexibility is penalized by $n-d-1$.
- Unlike Cp, AIC and BIC, for which a small value indicates a model with a low test error, a **large** value of adjusted $R^2$ indicates a model with lower test error.

Mallow's $C_p$:

$$C_p=\frac1n(RSS+2d\hat{\sigma}^2)$$
$\hat{\sigma}^2$ is an estimate of the variance of the error $\epsilon$, $d$ is the number of predictors.

AIC:

$$AIC=\frac1{n\hat{\sigma}^2}(RSS+2d\hat{\sigma}^2)$$
- AIC criterion is defined for a large class of models fit by maximum likelihood

BIC:

$$BIC=\frac1n(RSS+\log(n)d\hat{\sigma}^2)$$

- Validation and cross-validation
	- Such procedure has an advantage over metrics like AIC, BIC, etc., it provides **a direct estimate of the test error, and makes fewer assumptions about the true underlying model.**

### Regularization Shrinkage Methods

- Shrinkages methods: **LASSO and Ridge**
- Instead of fitting a linear model that contains a subset of the predictors, we can fit a model with all *p* predictors using a **technique that constrains or regularizes the coefficient estimates, or to say that shrinks the coefficients towards zero.**
	- the constraint -> shrink the coefficients ->  reduce the variance significantly
- ==**Ridge regression**==
	- Introducing the ridge regression coefficient estimates $\hat{\beta}^R$, are the values that minimize:
$$\sum^n_{i=1}\left(y_i-\beta_0-\sum^p_{j=1}\beta_jx_{ij}\right)^2+\lambda\sum^p_{j=1}\beta_j^2=RSS +\lambda\sum^p_{j=1}\beta_j^2$$

where $\lambda>0$ is a **tuning parameter** to be determined separately.

- Ridge regression -> seeks coefficient estimates that fit the data well by making the RSS small
	- However, the term $\lambda\sum^p_{j=1}\beta_j^2$ is a **shrinkage penalty**, it is small when all $\beta_j$s are small. So it has the effect of **shrinking** the estimates of $\beta_j$ towards zero.
	- The tuning parameter $\lambda$ -> controls the relative impact of these two terms on the regression coefficient estimates.
		- $\lambda=0$ -> no effect, ridge regression = normal linear regression with least squares estimates
		- $\lambda\to\infty$  the impact of penalty grows, and the ridge regression coefficient estimates -> zero
	- Ridge regression generates a set of $\hat{\beta}_\lambda^R$ for each value of $\lambda$.
- The ridge regression coefficient estimates can change *substantially* when multiplying a given predictor by a constant. ->
	- ==**It is best to apply ridge regression after standardizing the predictors**==

$$\hat{x}_{ij}=\frac{x_{ij}}{\sqrt{\frac1n\sum^n_{i=1}(x_{ij}-\bar{x}_j)^2}}$$

==Why does ridge regression improve over least squares?==

- Because of the **bias-variance trade-off**. As $\lambda$ increases, the flexibility of the ridge regression fit decreases (more constraints), leading to decreased variance but increase bias.
- In general, in situations where the relationship between the response and the predictors is close to linear, the least squares estimates will have low bias but may have high variance. -> A small change in the training data set -> large change in the least squares coefficient estimates.
- When the number of variables $p$ is almost as large as the number of observations $n$, the least squares estimates will be extremely variable.
	- If $p>n$, then the least squares estimates do not even have a unique solution.
	- But **ridge regression** can still perform well by trading off a small increase in bias for large decrease in variance. Hence, **ridge regression works best in situations where the least squares estimates have high variance.**
	- **Ridge regression also has substantial computational advantage over best subset selection ($2^p$ models).**

**LASSO**

- Ridge regression has one obvious disadvantage: it **will include all $p$ predictors in the final model.** The penalty $\lambda\sum\beta_j^2$ will shrink all of the coefficients towards zero, but it will not set any of them exactly to zero (unless $\lambda=\infty$).
	- not a problem for prediction accuracy, but it **creates a challenge in model interpretation in settings in which the number of variable $p$ is large.**

- The **lasso** is a relatively recent alternative to ridge regression that **overcomes this disadvantage** with the lasso coefficients $\hat{\beta}_\lambda^L$ minimize the quantity:

$$\sum^n_{i=1}\left(y_i-\beta_0-\sum^p_{j=1}\beta_jx_{ij}\right)^2+\lambda\sum^p_{j=1}\vert\beta_j\vert=RSS + \lambda\sum^p_{j=1}\vert\beta_j\vert$$

- Basically, lasso replaced the $\mathcal{l}_2$ penalty with $\mathcal{l}_1$ penalty, where $\vert\vert\beta\vert\vert_1=\sum\vert\beta_j\vert$
- Difference: $l_1$ penalty has the effect of forcing some of the coefficient estimates to be exactly equal to zero when the tuning parameter $\lambda$ is sufficiently large.
- -> Lasso performs "variable selection" -> sparse models -> more interpretable -> only a subset of variables.

> Contour: $l_1: \vert\beta_1\vert+\vert\beta_2\vert\leq s$, $l_2:\beta_1^2+\beta_2^2\leq s$

- The lasso is much more closely related to best subset selection, since the lasso performs feature selection for $s$ sufficiently small while ridge regression does not.

### ==Comparing the Lasso and Ridge Regression==

- Lasso has a major advantage over ridge regression -> it produces simpler and more interpretable models that involve only a subset of the predictors.
- But neither will universally dominate the other.
- In general,
	- ==The lasso performs better when a relatively small number of predictors have substantial coefficients, and the remaining predictors have coefficients that are very small or that equal zero.==
	- ==Ridge regression will perform better when the response is a function of many predictors, all with coefficients of roughly equal size.==
	- **However, the number of predictors that is related to the response is never known a priori for reality.**
		- We might use cross-validation to determine which approach is better on a particular data set.

### PCR and PLS

$Z_1, Z_2,\dots, Z_M$ represent $M<p$ linear combinations of the original $p$ predictors with

$$Z_m=\sum^p_{j=1}\phi_{jm}X_j$$ for some constants $\phi_{1m},\phi_{2m},\dots, \phi_{pm}$ where $m=1,\dots, M$.

So basically, this is projecting $p$-dimensional data on a $M$-dimensional plane.

All dimension reduction methods work in two steps:

1. The transformed predictors $Z_1, Z_2,\dots, Z_m$ are obtained.
2. The model is fit using $M$ predictors.

However, the choice of $Z_1,\dots,Z_M$ or equivalently the selection the $\phi_{jm}$'s can be achieved in different ways.

### **Principal Component Analysis (PCA)**

- =="deriving a low-dimensional set of features from a large set of variables"==
- zero correlation condition of $Z_1$ with $Z_2$ -> direction must be perpendicular/orthogonal
- The principal components regression (**PCR**) involves constructing the first $M$ principal components $Z_1,\dots, Z_m$ and then using these components as the predictors -> a linear regression with least squares fit.
- Key idea: **a small number of principal components suffice to explain most of the variability in the data, as well as the relationship with the response.**
- Ridge regression is a "continuous version of PCR."
- The number of principal components $M$ is typically chosen by cross-validation.
- It is recommended to **standardize** each predictor prior to generating the principal components.

### **Partial Least Squares (PLS)**

- PCR -> unsupervised -> PCR suffers from **a drawback: there is no guarantee that the directions that best explain the predictors will also be the best directions to use for predicting the response.**
	- As long as the data is sufficient, this should be fine.
	- But if some data with leverage (?) -> discrepancy

`TODO`

## Chapter 7 Moving Beyond Linearity

### Polynomial regression and step functions

- Some extensions of linear models
	- **Polynomial regression**: adding extra predictors -> raising each of the original predictors to a power.
	- **Step functions**: cut the range of a variable into $K$ distinct regions in order to produce a qualitative variable ==  fitting piecewise constant function.
	- **Regression splines**:
		- more flexible than polynomial and step functions
		- "extension of the two": 1. dividing the range of $X$ into $K$ distinct regions; 2. within each range, a polynomial function is fit to the data. But the polynomials are **constrained so that they join smoothly at the region boundaries, or ==knots==**.
		- very flexible
	- **Smoothing splines**: similar to regression splines, but resulting from minimizing a residual sum of squares criterion subject to a smoothness penalty.
	- **Local regression**: similar to splines, but differs in ==allowing regions to overlap==.
	- **Generalized additive models (GAM)**: the methods above but on multiple predictors.

- Polynomial Regression:
	- Usually we do order 3 or 4 at most. When the order is too high, we could have some very strange shapes and the curve can be overly flexible.
- Step Functions (aka **piecewise-constant regression model**)
	- non-global structure -> avoid imposing a global structure.
	- by breaking the range of $X$ into **bins**, and fit a different **constant** in each bin.
	- ==**converting a continuous variable into an ordered categorical variable.**==
	- “a collection of piecewise-constant functions”
- "**Basis functions**" -> both polynomial and piecewise-constant regression models are special cases of a *basis function.*

### Regression Splines and GAMs

- **Piecewise polynomials (not even splines yet)**:
	- Instead of fitting a high-degree polynomial over the entire range of $X$, **piecewise polynomial regression** -> fitting separate low-degree polynomials over different regions of $X$.
	- the problem is that it's not continuous, not smooth either.
- so here introduce the splines, also as constraints.
	- e.g. (piecewise cubic -> continuous piecewise cubic -> cubic spline)
- **The general definition of a degree-$d$ spline is ==it is a piecewise degree-$d$ polynomial, with continuity in derivatives up to degree $d-1$ at each knot.==**

- Splines have high variance at the outer range of the predictors, that is when the predictor is small or large.
- So another definition is the **natural splines** where additional constraints are introduced as the boundary of the regression spline. (narrower)

- How to find the number of knots? Use cross-validation to find the knots $K$ with the smallest RSS.

- **Comparison between polynomial regression and regression splines:** Regression splines often give superior results to polynomial regression (as the extra flexibility in the polynomial -> undesirable results at the boundaries.)

- **==Smoothing splines==**
	- For a fitted curve $g(x)$, we want $RSS=\sum^n_{i=1}(y_i-g(x_i))^2$ as small as possible.
		- Problem: if no constraints on $g(x_i)$, then we can choose $g$ to interpolate all of the $y_i$ so that $RSS=0$.
			- So we want $RSS$ small but also $g$ smooth.
	- How to ensure $g$ is smooth?
		- (natural approach) find the function $g$ that minimizes:
$$\sum^n_{i=1}(y_i-g(x_i))^2+\lambda\int g''(t)^2dt$$ where $\lambda$ is a nonnegative *tuning parameter* -> $g$ is called a **smoothing spline.**
- "Loss + Penalty"
	- roughly speaking, the second derivative of a function is a measure of its *roughness.*
		- The larger -> more wiggly.
	- The integral of the second derivative -> **a measure of the total change in the function $g'(t)$ (the first derivative) over the entire range.**
	- So the larger $\lambda$ -> the smoother $g$ will be. The penalty term encourages $g$ to be smooth, as $g$ smooth -> the integral will be small.
		- e.g. $\lambda=0$ -> penalty term has no effect, $g$ will be very jumpy.

- *effective degrees of freedom != degree of freedom*

- **Local regression**: fitting non-linear functions which involves computing the fit at a target point $x_0$ using only the nearby training observations.
	- A *memory-based* procedure, just like nearest-neighbors, we need *all the training data* each time we wish to compute a prediction.

- **==Generalized Additive Models (GAM)==** provides a general framework for extending a standard linear model by allowing non-linear functions of each of the variables, while maintaining **additivity.**
	- **Backfitting** is a method that fits a model involving multiple predictors by repeatedly updating the fit for each predictor in turn, holding the others fixed.
	- **Pros of GAM**
		- Allow to fit a non-linear $f_j$ to each $X_j$ -> standard linear regression will miss
		- The non-linear fits can potentially make more accurate predictions.
		- Additive, so we can examine the effect of each $X_j$ on $Y$ individually while holding all of the other variables fixed.
		- Smoothness of the function $f_j$ for the variable $X_j$ can be summarized via degrees of freedom.
	- **Cons of GAM**
		- restricted to be additive -> important interactions can be missed.
			- But we can also include additional predictors of the form $X_j\times X_k$ manually.
			- We can add low-dimensional interaction functions of the form $f_{jk}(X_j, X_k)$ into the model, such terms can be fit using two-dimensional smoothers such as local regression, or two-dimensional splines.

## Chapter 4 Classification

qualitative -> classification

- Two reasons not to perform classification using a regression method:
	- A regression method cannot accommodate a qualitative response with more than two classes;
	- a regression method will not provide meaningful estimates of $Pr(Y|X)$, even with just two classes.

### Logistic Regression (p.130-138)
- Rather than modeling the response, logistic regression models the *probability that $Y$ belongs to a particular category,*  given a particular setup of predictor(s), e.g.
	- $$p(X) = Pr(Y=1|X)$$
- Model the probability $p(X)$ between 0 and 1 with the **logistic function**:
	- $$p(X)=\frac{\exp (\beta_0+\beta_1X)}{1+\exp(\beta_0+\beta_1 X)}$$
- "maximum likelihood"
- S-Shaped -> no matter what value $X$ is, the prediction is very **sensible**
- $$\frac{p(X)}{1-p(X)}=e^{\beta_0+\beta_1 X}$$
	- This quantity is also called the **odds** from 0 to $\infty$
	- If we take log on both sides, we get: $$\log\left(\frac{p(X)}{1-p(X)}\right)=\beta_0+\beta_1 X$$
	- The left-hand side is also called **logit** or **log odds**
- **maximum likelihood**: seek estimates for $\beta_0$ and $\beta_1$ such that the predicted probability $\hat{p}(x_i)$ of response for each individual, corresponds as closely as possible to the individual's observed response values.
	- In other words, if we plug in the data, the predicted two categories should be as close as 1 or 0.
- For multiple logistic regression:
	- $$\log\left(\frac{p(X)}{1-p(X)}\right)=\beta_0+\beta_1X_1+\cdots +\beta_pX_p$$
	- rewritten as $$p(X)=\frac{e^{\beta_0+\beta_1X_1+\cdots +\beta_pX_p}}{1+e^{\beta_0+\beta_1X_1+\cdots +\beta_pX_p}}$$
	- Then we use maximum likelihood method to find $\beta_0, \beta_1,\dots, \beta_p$
- While the previous method only fits the case where $K=2$ as a binary classifier, we can do something as an extension with $K>2$ , this is called **multinomial logistic regression**
	- $$Pr(Y=k|X=x)=\frac{e^{\beta_{k0}+\beta_{k1}X_1+\cdots +\beta_{kp}X_p}}{1+\sum^{K-1}_{l=1}e^{\beta_{l0}+\beta_{l1}X_{l1}+\cdots +\beta_{lp}X_{lp}}}$$ for $k=1, \dots, K-1$, and $$Pr(Y=K|X=x)=\frac{1}{1+\sum^{K-1}_{l=1}e^{\beta_{l0}+\beta_{l1}X_{l1}+\cdots +\beta_{lp}X_{lp}}}$$, naturally, we have $$\log\left(\frac{Pr(Y=k|X=x)}{Pr(Y=K|X=x)}\right)=\beta_{k0}+\beta_{k1}x_1+\cdots +\beta_{kp}x_p$$
- Another alternative coding for multinomial logistic regression is the **softmax coding**:
	- $$Pr(Y=k|X=x)=\frac{e^{\beta_{k0}+\beta_{k1}X_1+\cdots +\beta_{kp}X_p}}{\sum^{K}_{l=1}e^{\beta_{l0}+\beta_{l1}X_{l1}+\cdots +\beta_{lp}X_{lp}}}$$
- But it has some disadvantages:
	- When there is substantial separation between two classes -> parameter estimates for the logistic regression are *unstable*
	- If the distribution of the predictors $X$ is roughly normal in each of the classes and the sample size is small, the logistic regression could be not accurate
- We might need to look at some other classification methods (which of course supports multinomial cases)

### KNN (p.37-42, p.129-130)
### KNN regression (p.104-109)
- **KNN regression**
- Non-parametric methods do not explicitly assume a parametric form for $f(X)$ -> flexible approach for regression
	- **KNN regression** -> closely related to the KNN classifier
		- Given a value for $K$ and a prediction point $x_0$, KNN regression first identifies the $K$ training observations that are closest to $x_0$, represented by $\mathcal{N}_0$ -> It then estimates $f(x_0)$ using the average of all the training responses in $\mathcal{N}_0$.
		- In other words: $$\hat{f}(x_0)=\frac1K\sum_{x_i\in \mathcal{N}_0}y_i$$
		- Basically, KNN-regression votes for the predicted value as a mean.
		- In general, the optimal $K$ depends on the bias-variance tradeoff.
			- Small $K$ -> flexible -> low bias -> high variance (prediction depends on few points)
			- Large $K$ -> smoother and less variable fit -> low variance high bias (prediction made by a region, masking some of the structure in $f(X)$)

### Discriminant Analysis (p.138-150)

- For $p=1$ only one predictor, assume $f_k(x)$ is normal or Gaussian with $\mu_k$ and $\sigma_k^2$ are mean and variance for the $k$th class, $\sigma^2$ is the common shared variance across all $K$ classes.

$$p_k(x)=\frac{\pi_k\frac1{\sqrt{2\pi}\sigma}\exp\left(-\frac1{2\sigma^2}(x-\mu_k)^2\right)}{\sum^K_{l=1}\pi_l\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac1{2\sigma^2}(x-\mu_l)^2\right)}$$ $\pi_k$ is the prior probability that an observation belongs to the $k$th class.

Then we **take the log**:

$$\delta_k(x)=x\cdot\frac{\mu_k}{\sigma^2}-\frac{\mu_k^2}{2\sigma^2}+\log(\pi_k)$$
**assign the observation to the class for which this value is the largest!**

 The **==Linear Discriminant Analysis (LDA)==** method approximates the Bayes classifier by plugging estimates for $\pi_k, \mu_k, \sigma^2$ into the equation above. In particular, the following estimates are used:

- $\hat{\mu}_k=\frac1{n_k}\sum_{i:y_i=k}x_i$
- $\hat{\sigma}^2=\frac1{n-K}\sum^K_{k=1}\sum_{i:y_i=k}(x_i-\hat{\mu}_k)^2$
- where $n$ is the total number of training observations, $n_k$ is the number of training observations in the $k$th class. $\mu_k$ is the average of all the training observations from the $k$th class, $\hat{\sigma}^2$ is a weighted average of the sample variances for each of the $K$ classes. If no prior knowledge is known, $\hat{\pi_k} =n_k/n$ for simplicity.

Then LDA classifier plugs the estimates into the equation and assigns an observation $X=x$ to the class for which $$\hat{\delta}_k(x)=x\cdot\frac{\hat{\mu}_k}{\hat{\sigma}^2}-\frac{\hat{\mu}_k^2}{2\hat{\sigma}^2}+\log(\hat{\pi}_k)$$ is the largest. Since the discriminant functions of $x$ are **linear**, we call it LDA.

- This can be extended to multinomial Gaussian -> case $p>1$.

### Classification Wrap-up (p.151-154)

- **sensitivity** and **specificity**

- True class
	- -
		- predicted -: True Neg (TN)
		- predicted +: False Pos (FP)
	- +
		- predicted -: False Neg (FN)
		- predicted +: True Pos (TP)
- False Pos. rate: FP/(TN+FP) == **Type I error, 1-Specificity**
- True Pos. rate: TP/(TP+FN) == **1 - Type II error, power, sensitivity, recall**
- Pos. Pred. value: TP/(TP+FP) == **Precision, 1-false discovery proportion**
- Neg. Pred. value: TN/(TN+FN)

- **naïve Bayes classifier**
	- Naive Bayes classifier estimates $f_1(x), \dots, f_K(x)$ by not assuming these functions belong to a particular family of distributions (like multivariate normal), but rather assume:
		- $$f_k(x)=f_{k1}(x_1)\times f_{k2}(x_2)\times\cdots\times f_{kp}(x_p)$$
		- This is simplifying the whole situation by getting rid of all the joint distributions.
	- So we have:
		- $$Pr(Y=k|X=x)=\frac{\pi_k\cdot f_{k1}(x_1)\cdots f_{kp}(x_p)}{\sum^K_{l=1}\pi_l\cdot f_{l1}(x_1)\cdots f_{lp}(x_p)}$$ for $k=1,\dots, K$.

- **A Comparison of Classification Methods**
	- LDA is a special case of QDA
	- any classifier with a linear decision boundary is a special case of naïve Bayes (LDA is a special case of naïve Bayes)
		- LDA assumes the features are normally distributed with a common within-class covariance matrix, and naïve Bayes assumes independence of the features.
	- KNN is an example of non-parametric, better performance when the decision boundary is highly non-linear
	- KNN requires a lot of observations relative to the number of predictors, $n \gg p$
	- KNN does not tell us which predictors are important.



## Chapter 8 Tree-Based Methods

## Chapter 10 Deep Learning

### Single Layer Neural Networks
### Multiple Layer Neural Networks