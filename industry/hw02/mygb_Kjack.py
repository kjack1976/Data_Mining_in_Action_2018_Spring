# Шаблон из скрипта gb_impl_example.py
#coding=utf-8

from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor
import numpy as np


# Параметрами с которыми вы хотите обучать деревья
TREE_PARAMS_DICT = {'max_depth':19}
# Параметр tau (learning_rate) для вашего GB
TAU = 0.0001



class SimpleGB(BaseEstimator):
    def __init__(self, tree_params_dict, iters, tau):
        self.tree_params_dict = tree_params_dict
        self.iters = iters
        self.tau = tau
        self.global_res = [] # введем переменную, что бы посмотреть, что  у нас собрает переменная res
        
    def fit(self, X_data, y_data):
        self.base_algo = DecisionTreeRegressor(**self.tree_params_dict).fit(X_data, y_data)
        self.estimators = []
        curr_pred = self.base_algo.predict(X_data)
        for iter_num in range(self.iters):
            # Нужно посчитать градиент функции потерь
            grad = -y_data*np.exp(-y_data*curr_pred)/(1 + np.exp(-y_data*curr_pred)) # TODO
            # Нужно обучить DecisionTreeRegressor предсказывать антиградиент
            # Не забудьте про self.tree_params_dict
            algo = DecisionTreeRegressor(**self.tree_params_dict).fit(X_data, - grad) # TODO

            self.estimators.append(algo)
            # Обновите предсказания в каждой точке
            curr_pred += self.estimators[-1].predict(X_data) # TODO
        return self
    
    def predict(self, X_data):
        # Предсказание на данных
        res = self.base_algo.predict(X_data)
        for estimator in self.estimators:
            res += self.tau * estimator.predict(X_data)
            
        # Задача классификации, поэтому надо отдавать 0 и 1
        # Мои примечания - Нужно попробовать применить логистическую ф-ю к res для определния класса 
        t = 0.5
        res_class = res
        #res_class = 1/(1+np.exp(-res)) # добавим сигмоид, что бы получить вероятности классов
        res_class[res_class > t] = 1 # если вероятность больше t=0.5, то класс 1
        res_class[res_class <= t] = 0 # в противном случае - класс 0
        
        return res_class

