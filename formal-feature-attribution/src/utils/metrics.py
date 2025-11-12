import numpy as np
from scipy.stats import kendalltau
from typing import Dict, Tuple


def calculate_correlations(ffa_attr: np.ndarray, lime_attr: np.ndarray, 
                          shap_attr: np.ndarray, perm_attr: np.ndarray) -> Dict[str, float]:
    """
    Calcula correlações entre diferentes métodos de atribuição
    
    Args:
        ffa_attr: Atribuição FFA formal
        lime_attr: Atribuição LIME
        shap_attr: Atribuição SHAP
        perm_attr: Importância por permutação
        
    Returns:
        Dict com correlações entre métodos
    """
    correlations = {}
    
    try:
        corr, _ = kendalltau(ffa_attr, lime_attr)
        correlations['ffa_vs_lime'] = corr if not np.isnan(corr) else 0.0
        
        corr, _ = kendalltau(ffa_attr, shap_attr)
        correlations['ffa_vs_shap'] = corr if not np.isnan(corr) else 0.0
        
        corr, _ = kendalltau(ffa_attr, perm_attr)
        correlations['ffa_vs_perm'] = corr if not np.isnan(corr) else 0.0
        
        corr, _ = kendalltau(lime_attr, shap_attr)
        correlations['lime_vs_shap'] = corr if not np.isnan(corr) else 0.0
        
    except Exception as e:
        correlations = {
            'ffa_vs_lime': 0.0,
            'ffa_vs_shap': 0.0, 
            'ffa_vs_perm': 0.0,
            'lime_vs_shap': 0.0
        }
    
    return correlations


def calculate_ranking_metrics(ffa_attr: np.ndarray, lime_attr: np.ndarray,
                             shap_attr: np.ndarray, perm_attr: np.ndarray) -> Dict[str, any]:
    """
    Calcula métricas de ranking e top features
    
    Args:
        ffa_attr: Atribuição FFA formal
        lime_attr: Atribuição LIME  
        shap_attr: Atribuição SHAP
        perm_attr: Importância por permutação
        
    Returns:
        Dict com métricas de ranking
    """
    metrics = {}
    
    methods = {
        'ffa': ffa_attr,
        'lime': lime_attr,
        'shap': shap_attr,
        'perm': perm_attr
    }
    
    top_features = {}
    for method_name, attr in methods.items():
        if len(attr) > 0:
            top_idx = np.argmax(attr)
            top_features[method_name] = {
                'feature_index': top_idx,
                'score': attr[top_idx]
            }
    
    metrics['top_features'] = top_features
    
    top_indices = [top['feature_index'] for top in top_features.values()]
    agreement = len(set(top_indices)) == 1  
    
    metrics['top_feature_agreement'] = agreement
    metrics['unique_top_features'] = len(set(top_indices))
    
    return metrics


def calculate_manhattan_distance(attr1: np.ndarray, attr2: np.ndarray) -> float:
    """
    Calcula distância Manhattan entre vetores de atribuição
    
    Args:
        attr1: Primeiro vetor de atribuição
        attr2: Segundo vetor de atribuição
        
    Returns:
        float: Distância Manhattan
    """
    return np.sum(np.abs(attr1 - attr2))