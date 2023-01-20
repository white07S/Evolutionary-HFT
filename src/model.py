import pandas as pd
import numpy as np
import json
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMClassifier
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from data_pipeline import check_null, preprocessing, fill_null,x_y_split,correlation_filter
from feature_eng import feature_eng,feature_selection,features
class model:
    lgbm_paramgrid = {
        'learning_rate': np.arange(0.0005, 0.0015, 0.0001),
        'n_estimators': range(800, 2000, 200),
        'max_depth': [3, 4],
        'colsample_bytree': np.arange(0.2, 0.5, 0.1),
        'reg_alpha': [1],
        'reg_lambda': [1]
    }

    @staticmethod
    def random_forest(x, y):
        rf = RandomForestClassifier(n_estimators=200, max_depth=8)
        rf.fit(x, y)
        return rf

    @classmethod
    def lightgbm(cls, x, y):
        keys, vals = list(zip(*cls.lgbm_paramgrid.items()))
        products = list(product(*vals))
        param_comb = [dict(zip(keys, i)) for i in products]

        if len(param_comb) > 1000:
            best_param = cls.GA_tune_lgbm(x, y)
        else:
            best_param = cls.GS_tune_lgbm(x, y)

        lightgbm = LGBMClassifier(**best_param)
        lightgbm.fit(x, y)

        return lightgbm

    @classmethod
    def GA_tune_lgbm(cls, x, y):
        tuner = EvolutionaryAlgorithmSearchCV(
            estimator=LGBMClassifier(),
            params=cls.lgbm_paramgrid,
            scoring="accuracy",
            cv=TimeSeriesSplit(n_splits=4),
            verbose=1,
            population_size=50,
            gene_mutation_prob=0.2,
            gene_crossover_prob=0.5,
            tournament_size=3,
            generations_number=20,
        )
        tuner.fit(x, y)
        return tuner.best_params_

    @classmethod
    def GS_tune_lgbm(cls, x, y):
        tuner = GridSearchCV(
            estimator=LGBMClassifier(),
            param_grid=cls.lgbm_paramgrid,
            scoring="accuracy",
            cv=TimeSeriesSplit(n_splits=4),
            verbose=1,
            n_jobs=-1,
        )
        tuner.fit(x, y)
        return tuner.best_params_


class feature:
    @staticmethod
    def save(features, correlation_remove):
        final = {
            'keep_features': features,
            'correlation_remove': correlation_remove
        }

        with open('features.txt', 'w') as f:
            f.write(json.dumps(final))

    @staticmethod
    def load(features):
        features = json.dumps(features)
        return features


def train_model(data, target_label):
    data = data.copy()
    data = preprocessing(data)
    check_null(data)
    data = fill_null(data)
    x, y = x_y_split(data)
    x = feature_eng.basic_features(x)
    x = correlation_filter.filter(x)
    x = feature_eng.lag_rolling_features(x)
    x, y = feature_eng.remove_na(x, y)
    y = y[target_label]
    features = feature_selection.select(x, y)
    feature.save(features, correlation_filter.remove_cols)
    lightgbm = model.lightgbm(x[features], y)
    rf = model.random_forest(x[features], y)
    joblib.dump(rf, 'rf.joblib')
    joblib.dump(lightgbm, 'lgbm.joblib')


def predict(data, target_label,features):
    '''returns both the prediction and the target_label'''
    features = feature.load(features)['keep_features']
    correlation_remove = feature.load(features)['correlation_remove']
    data = data.copy()
    data = preprocessing(data)
    data = fill_null(data)
    x, y = x_y_split(data)
    x = feature_eng.basic_features(x)
    x = x.drop(correlation_remove, axis=1)
    x = feature_eng.lag_rolling_features(x)
    x, y = feature_eng.remove_na(x, y)
    y = y[target_label]
    x = x[features]
    lgbm = joblib.load('lgbm.joblib')
    rf = joblib.load('rf.joblib')
    lgbm_predict = lgbm.predict_proba(x)
    rf_predict = rf.predict_proba(x)
    final_predict = (lgbm_predict + rf_predict) / 2
    final_predict = np.argmax(final_predict, axis=1)

    return final_predict, y


