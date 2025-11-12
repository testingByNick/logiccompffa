import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
import lime
import lime.lime_tabular
from typing import List, Optional
import warnings


class ApproximationMethods:
    """
    Implementa métodos de aproximação para comparação com FFA:
    - LIME
    - SHAP (aproximação)
    - Permutation Importance
    """
    
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.n_features = len(feature_names)
    
    def permutation_importance(self, X: np.ndarray, y: np.ndarray, 
                             n_repeats: int = 10) -> np.ndarray:
        """
        Calcula importância por permutação
        
        Args:
            X: Dados de entrada
            y: Labels
            n_repeats: Número de repetições
            
        Returns:
            np.ndarray: Scores de importância normalizados
        """
        try:
            # Implementação simplificada para controle
            baseline_score = accuracy_score(y, self.model.predict(X))
            importances = np.zeros(self.n_features)
            
            for i in range(self.n_features):
                X_permuted = X.copy()
                # Permuta a feature i
                X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
                permuted_score = accuracy_score(y, self.model.predict(X_permuted))
                importances[i] = max(0, baseline_score - permuted_score)
            
            # Normaliza
            if np.sum(importances) > 0:
                importances = importances / np.sum(importances)
            else:
                importances = np.ones(self.n_features) / self.n_features
                
            return importances
            
        except Exception as e:
            warnings.warn(f"Erro em permutation importance: {e}")
            return np.ones(self.n_features) / self.n_features
    
    def lime_attribution(self, X: np.ndarray, instance_idx: int, 
                        num_features: Optional[int] = None) -> np.ndarray:
        """
        Calcula atribuição usando LIME
        
        Args:
            X: Dados de entrada
            instance_idx: Índice da instância a explicar
            num_features: Número de features para incluir na explicação
            
        Returns:
            np.ndarray: Scores de atribuição do LIME
        """
        if num_features is None:
            num_features = self.n_features
            
        try:
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X,
                feature_names=self.feature_names,
                class_names=['class_0', 'class_1'],
                mode='classification',
                random_state=42
            )
            
            instance = X[instance_idx]
            exp = explainer.explain_instance(
                instance,
                self.model.predict_proba,
                num_features=num_features
            )
            
            # Converte para array de scores
            lime_scores = np.zeros(self.n_features)
            for feature, score in exp.as_list():
                for i, name in enumerate(self.feature_names):
                    if name in feature:
                        lime_scores[i] = abs(score)
                        break
            
            # Normaliza
            if np.sum(lime_scores) > 0:
                lime_scores = lime_scores / np.sum(lime_scores)
            else:
                lime_scores = np.ones(self.n_features) / self.n_features
            
            return lime_scores
            
        except Exception as e:
            warnings.warn(f"Erro no LIME: {e}")
            return np.ones(self.n_features) / self.n_features
    
    def shap_approximation(self, X: np.ndarray, instance_idx: int, 
                          num_samples: int = 50) -> np.ndarray:
        """
        Aproximação do SHAP baseada em valores de baseline
        
        Args:
            X: Dados de entrada
            instance_idx: Índice da instância a explicar
            num_samples: Número de amostras para cálculo
            
        Returns:
            np.ndarray: Scores de atribuição aproximados
        """
        try:
            baseline = np.median(X, axis=0)
            instance = X[instance_idx]
            original_pred = self.model.predict_proba([instance])[0, 1]
            attributions = np.zeros(self.n_features)
            
            for i in range(self.n_features):
                total_effect = 0
                valid_samples = 0
                
                for _ in range(num_samples):
                    # Cria instância híbrida
                    hybrid_instance = instance.copy()
                    hybrid_instance[i] = baseline[i]
                    
                    try:
                        hybrid_pred = self.model.predict_proba([hybrid_instance])[0, 1]
                        effect = abs(original_pred - hybrid_pred)
                        total_effect += effect
                        valid_samples += 1
                    except:
                        continue
                
                if valid_samples > 0:
                    attributions[i] = total_effect / valid_samples
                else:
                    attributions[i] = 0.0
            
            # Normaliza
            if np.sum(attributions) > 0:
                attributions = attributions / np.sum(attributions)
            else:
                attributions = np.ones(self.n_features) / self.n_features
            
            return attributions
            
        except Exception as e:
            warnings.warn(f"Erro no SHAP: {e}")
            return np.ones(self.n_features) / self.n_features