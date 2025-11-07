import numpy as np
from z3 import Solver, Real
from typing import List, Set, Tuple
import warnings


class FormalFFA:
    """
    Baseado na Definição 1: AXp (Abductive Explanations)
    """
    
    def __init__(self, model, feature_bounds: List[Tuple[float, float]]):
        self.model = model
        self.feature_bounds = feature_bounds
        self.n_features = len(feature_bounds)
    
    def check_axp(self, instance: np.ndarray, feature_subset: Set[int], target: int, 
                  n_samples: int = 100) -> bool:
        """
        Args:
            instance: Instância a ser explicada
            feature_subset: Conjunto de índices de features
            target: Classe alvo
            n_samples: Número de amostras para verificação
            
        Returns:
            bool: True se o subconjunto é uma AXp
        """
        try:
            for _ in range(n_samples):
                test_instance = instance.copy()
                
                for i in range(self.n_features):
                    if i not in feature_subset:
                        low, high = self.feature_bounds[i]
                        test_instance[i] = np.random.uniform(low, high)
                
                prediction = self.model.predict([test_instance])[0]
                if prediction != target:
                    return False
            
            return True
            
        except Exception as e:
            warnings.warn(f"Erro na verificação AXp: {e}")
            return False

    def enumerate_axps_marco(self, instance: np.ndarray, target: int) -> List[Set[int]]:
    """
    Implementa o algoritmo MARCO do artigo para enumeração eficiente de AXp's
    """
    # TODO: Implementar algoritmo de hitting set duality
    pass
    
    def compute_ffa(self, instance: np.ndarray, target: int, 
                   n_combinations: int = 50) -> np.ndarray:
        """
        Args:
            instance: Instância a ser explicada
            target: Classe alvo
            n_combinations: Número de combinações para testar
            
        Returns:
            np.ndarray: Scores de atribuição para cada feature
        """
        scores = np.zeros(self.n_features)
        tested_subsets = 0
        
        for _ in range(n_combinations):
            subset_size = np.random.randint(1, self.n_features + 1)
            subset = set(np.random.choice(self.n_features, subset_size, replace=False))
            
            if self.check_axp(instance, subset, target, n_samples=20):
                for feature in subset:
                    scores[feature] += 1
                tested_subsets += 1
        
        if tested_subsets > 0:
            scores = scores / tested_subsets
        
        return scores


class HeuristicFFA:
    """
    Implementação heurística do FFA baseada em perturbações locais
    """
    
    def __init__(self, model, feature_names: List[str], n_features: int):
        self.model = model
        self.feature_names = feature_names
        self.n_features = n_features
    
    def compute(self, instance: np.ndarray, target: int, 
                epsilon: float = 0.1, n_samples: int = 100) -> np.ndarray:
        """
        Args:
            instance: Instância a ser explicada
            target: Classe alvo
            epsilon: Magnitude da perturbação
            n_samples: Número de amostras por feature
            
        Returns:
            np.ndarray: Scores de atribuição normalizados
        """
        original_prob = self.model.predict_proba([instance])[0, target]
        scores = np.zeros(self.n_features)
        
        for i in range(self.n_features):
            total_effect = 0
            
            for _ in range(n_samples):
                perturbed_instance = instance.copy()
                perturbation = epsilon * (2 * np.random.random() - 1)
                perturbed_instance[i] = np.clip(perturbed_instance[i] + perturbation, 0, 1)
                
                perturbed_prob = self.model.predict_proba([perturbed_instance])[0, target]
                total_effect += abs(original_prob - perturbed_prob)
            
            scores[i] = total_effect / n_samples
        
        if np.sum(scores) > 0:
            scores = scores / np.sum(scores)
        
        return scores
