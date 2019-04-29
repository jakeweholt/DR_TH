## LendingClub Loan Default Classifier
-------

### Model Versions

**Logistic Regression:**

Model Name: logistic_regression_model_1

Version: `0.1.1556570371`

Definition:
```
SGDClassifier(loss='log', penalty="l2", n_iter=1000)
```

Model Name: logistic_regression_model_2

Version: `0.1.1556570505`

Definition:
```
SGDClassifier(loss='log', n_iter=1000)
```

**XGBoost:**

**Name**: xgboost_model_1

**Version**: `0.1.1556570382`

**Model Definition**:
```
XGBClassifier(n_estimators=100, scale_pos_weight=6.77)
```

**Name**: xgboost_model_2

**Version**: `0.1.1556570523`

**Model Definition**:
```
XGBClassifier(n_estimators=100)
```


#### Model Validation

#### Creating New Models
