## LendingClub Loan Default Classifier

This is a model library for running and deploying models based on the LendingClub Loan Defualt data, which can be found here: https://www.lendingclub.com/info/download-data.action. Models here were trained on a subset of this data.

### Companion Docs/Notebooks
- [Business motivation and intro doc](https://docs.google.com/document/d/1V5CiQwuySPbKlDvfX8TpLxi0pXHN26-ehbBut4Noblc/edit?usp=sharing) (may require granted access).
- [Codebase commentary](https://docs.google.com/document/d/1LpQ2jej05sPmCyDdtWpO6lI0z7dLOUxuDXVyR5YXJKc/edit?usp=sharing)


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
