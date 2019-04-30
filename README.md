## LendingClub Loan Default Classifier

This is a model library for running and deploying models based on the LendingClub Loan Default data, which can be found here: https://www.lendingclub.com/info/download-data.action. Models here were trained on a subset of this data.

### First Order Tasks
1. Partition your data into a [holdout set and 5 stratified CV folds](https://github.com/jakeweholt/DR_TH/blob/a2e338bff058b2f1ece09e153b83def85d4bb6f1/model/model.py#L23-L70).
2. Pick any two machine learning algorithms from the list below, and build a binary classification model with each of them:
    - [Regularized Logistic Regression](https://github.com/jakeweholt/DR_TH/blob/c27aba7d213097f43029e91f69c9bc5d7bc0aa81/train_model.py#L42-L50)
    - [Gradient Boosting Machine](https://github.com/jakeweholt/DR_TH/blob/c27aba7d213097f43029e91f69c9bc5d7bc0aa81/train_model.py#L28-L40)
3. Both of your models must make use of numeric, categorical, text, and date features.
    - [Commentary on which columns are used](https://github.com/jakeweholt/DR_TH/blob/master/analysis/01_data_cleaning.ipynb).
4. Compute out-of-sample LogLoss and F1 scores on cross-validation and holdout.
    - [Current production model notebook](https://github.com/jakeweholt/DR_TH/blob/master/logistic_regression_model_validation_1.ipynb). 
5. Which one of your two models would you recommend to deploy? Explain your decision.
    - [Problem intro, business motivation and model selection doc](https://docs.google.com/document/d/1V5CiQwuySPbKlDvfX8TpLxi0pXHN26-ehbBut4Noblc/edit?usp=sharing) (may require granted access).
6. (Advanced, optional) Which 3 features are the most impactful for your model? Explain
your methodology.

### Companion Docs/Notebooks
- [Problem Intro, business motivation and model selection doc](https://docs.google.com/document/d/1V5CiQwuySPbKlDvfX8TpLxi0pXHN26-ehbBut4Noblc/edit?usp=sharing) (may require granted access).
- [Codebase commentary](https://docs.google.com/document/d/1LpQ2jej05sPmCyDdtWpO6lI0z7dLOUxuDXVyR5YXJKc/edit?usp=sharing)
- [Raw data EDA notebook](https://github.com/jakeweholt/DR_TH/blob/master/analysis/00_raw_data_EDA.ipynb)
- [Data cleaning notebook](https://github.com/jakeweholt/DR_TH/blob/master/analysis/01_data_cleaning.ipynb)

### Current Version in Production
- Name: logistic_regression_model_1, 
- Version: 0.1.1556570371
  - `SGDClassifier(loss='log', penalty="l2", n_iter=1000)`
- [Validation notebook](https://github.com/jakeweholt/DR_TH/blob/master/logistic_regression_model_validation_1.ipynb)

### Other Model Versions

**Logistic Regression**

- Name: logistic_regression_model_1, 
- Version: 0.1.1556570371
  - `SGDClassifier(loss='log', penalty="l2", n_iter=1000)`
- [Validation notebook](https://github.com/jakeweholt/DR_TH/blob/master/logistic_regression_model_validation_1.ipynb)
- Name: logistic_regression_model_2
- Version: 0.1.1556570505
  - `SGDClassifier(loss='log', n_iter=1000)`
- [Validation notebook](https://github.com/jakeweholt/DR_TH/blob/master/logistic_regression_model_validation_2.ipynb)

**XGBoost**

- Name: xgboost_model_1
- Version: 0.1.1556570382
  - `XGBClassifier(n_estimators=100, scale_pos_weight=6.77)`
- [Validation notebook](  https://github.com/jakeweholt/DR_TH/blob/master/xgboost_model_validation_1.ipynb)<br/>
- Name: xgboost_model_2
- Version: 0.1.1556570523
  - `XGBClassifier(n_estimators=100)`
- [Validation notebook](  https://github.com/jakeweholt/DR_TH/blob/master/xgboost_model_validation_2.ipynb)

#### Development

This was built using `Python 3.7.1`

Setup a dev environment of your choosing. Run `pip install -r requirements.txt`, which will install all necessary libraries.
