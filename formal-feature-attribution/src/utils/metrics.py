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

