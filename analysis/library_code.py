from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold


class XGBoostModelData:
    
    def __init__(self, X, y, test_size=0.15, random_state=1234):
        self.test_size = test_size
        self.random_state = random_state      
        self.X_raw = X
        self.y_raw = y
        self.transform()
        self.X_train, self.X_validate, self.y_train, self.y_validate = train_test_split(self.X_transformed, 
                                                                                y, 
                                                                                test_size=test_size, 
                                                                                stratify=y, 
                                                                                random_state=random_state)
        self._create_k_folds(k=5)
        
    def transform(self):
        self.X_transformed = self._create_categorical_dummies()
        
    def _create_categorical_dummies(self):
        df_transform = self.X_raw.copy()
        
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        categorical_columns = df_transform.select_dtypes(exclude=numerics)
        
        for col in categorical_columns:
            dummies = pd.get_dummies(categorical_columns[col])
            df_transform = df_transform.drop(col, axis=1)
            df_transform = pd.concat([df_transform, dummies], axis=1)
            
        return df_transform
    
    def _create_k_folds(self, k=5):
        skf = StratifiedKFold(n_splits=k, random_state=self.random_state, shuffle=True)
        i = 0
        self.k_folds = {}
        for train, test in skf.split(self.X_train, self.y_train):
            self.k_folds['fold_{}'.format(i)] = {
                'X_train':self.X_train.iloc[train],
                'y_train':self.y_train.iloc[train],
                'X_test':self.X_train.iloc[test],
                'y_test':self.y_train.iloc[test],
            }
            i+=1