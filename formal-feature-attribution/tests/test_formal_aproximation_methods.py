import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from approximation_methods import ApproximationMethods
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


class TestApproximationMethods(unittest.TestCase):
    
    def setUp(self):
        """Configuração para os testes"""
        X, y = make_classification(
            n_samples=100, n_features=4, n_informative=2,
            n_redundant=0, random_state=42
        )
        
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(X, y)
        
        self.feature_names = [f'F{i}' for i in range(4)]
        self.approx_methods = ApproximationMethods(self.model, self.feature_names)
        
        self.X = X
        self.y = y