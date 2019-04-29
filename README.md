## LendingClub Loan Default Classifier

This is a model library for running and deploying models based on the LendingClub Loan Defualt data, which can be found here: https://www.lendingclub.com/info/download-data.action. Models here were trained on a subset of this data.

### First Order Tasks
1. Partition your data into a [holdout set and 5 stratified CV folds](https://github.com/jakeweholt/DR_TH/model/model.py#L23-L70).
2. Pick any two machine learning algorithms from the list below, and build a binary classification model with each of them:
  - Regularized Logistic Regression (scikit-learn)
  - Gradient Boosting Machine (scikit-learn, XGBoost or LightGBM)
3. Both of your models must make use of numeric, categorical, text, and date features.
4. Compute out-of-sample LogLoss and F1 scores on cross-validation and holdout.
5. Which one of your two models would you recommend to deploy? Explain your decision.
6. (Advanced, optional) Which 3 features are the most impactful for your model? Explain
your methodology.

### Companion Docs/Notebooks
- [Business motivation and intro doc](https://docs.google.com/document/d/1V5CiQwuySPbKlDvfX8TpLxi0pXHN26-ehbBut4Noblc/edit?usp=sharing) (may require granted access).
- [Codebase commentary](https://docs.google.com/document/d/1LpQ2jej05sPmCyDdtWpO6lI0z7dLOUxuDXVyR5YXJKc/edit?usp=sharing)
- [Raw data EDA notebook](https://github.com/jakeweholt/DR_TH/blob/master/analysis/00_raw_data_EDA.ipynb)
- [Data cleaning notebook](https://github.com/jakeweholt/DR_TH/blob/master/analysis/01_data_cleaning.ipynb)

### Current Version in Production
- Name: logistic_regression_model_1, 
- Version: 0.1.1556570371
  - `SGDClassifier(loss='log', penalty="l2", n_iter=1000)`

### Other Model Versions

**Logistic Regression**

- Name: logistic_regression_model_1, 
- Version: 0.1.1556570371
  - `SGDClassifier(loss='log', penalty="l2", n_iter=1000)`
- Name: logistic_regression_model_2
- Version: 0.1.1556570505
  - `SGDClassifier(loss='log', n_iter=1000)`

**XGBoost**

- Name: xgboost_model_1
- Version: 0.1.1556570382
  - `XGBClassifier(n_estimators=100, scale_pos_weight=6.77)`
- Name: xgboost_model_2
- Version: 0.1.1556570523
  - `XGBClassifier(n_estimators=100)`


#### Model Validation

#### Creating New Models
