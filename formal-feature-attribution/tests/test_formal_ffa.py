import unittest
import numpy as np
import sys
import os

# Adiciona src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from formal_ffa import FormalFFA, HeuristicFFA
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


class TestFormalFFA(unittest.TestCase):
    
    def setUp(self):
        """Configuração para os testes"""
        # Dataset simples para testes
        X, y = make_classification(
            n_samples=100, n_features=4, n_informative=2, 
            n_redundant=0, random_state=42
        )
        X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)
        
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(X, y)
        
        self.feature_bounds = [(0, 1)] * 4
        self.formal_ffa = FormalFFA(self.model, self.feature_bounds)
        
        self.heuristic_ffa = HeuristicFFA(
            self.model, [f'F{i}' for i in range(4)], 4
        )
    
    def test_formal_ffa_initialization(self):
        """Testa inicialização do FormalFFA"""
        self.assertIsNotNone(self.formal_ffa)
        self.assertEqual(self.formal_ffa.n_features, 4)
        self.assertEqual(len(self.formal_ffa.feature_bounds), 4)
    
    def test_heuristic_ffa_initialization(self):
        """Testa inicialização do HeuristicFFA"""
        self.assertIsNotNone(self.heuristic_ffa)
        self.assertEqual(self.heuristic_ffa.n_features, 4)
        self.assertEqual(len(self.heuristic_ffa.feature_names), 4)
    
