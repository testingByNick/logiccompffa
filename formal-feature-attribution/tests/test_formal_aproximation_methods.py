import unittest
import numpy as np
import sys
import os

# Adiciona src ao path
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
    
    def test_permutation_importance(self):
        """Testa cálculo de importância por permutação"""
        importance = self.approx_methods.permutation_importance(self.X, self.y, n_repeats=5)
        
        self.assertEqual(importance.shape, (4,))
        self.assertTrue(np.all(importance >= 0))
        self.assertTrue(np.all(importance <= 1))
        # Verifica normalização
        self.assertAlmostEqual(np.sum(importance), 1.0, places=5)
    
    

if __name__ == '__main__':
    unittest.main()