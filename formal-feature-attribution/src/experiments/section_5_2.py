import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from approximation_methods import ApproximationMethods
from utils.metrics import calculate_correlations, calculate_ranking_metrics


def run_section_5_2(results_5_1):
    """
    Seção 5.2: Comparação de Métodos
    
    Args:
        results_5_1: Resultados da seção 5.1
        
    Returns:
        dict: Resultados da comparação
    """
    print("\n" + "=" * 70)
    print("🔍 SEÇÃO 5.2: COMPARAÇÃO COM MÉTODOS DE APROXIMAÇÃO")
    print("=" * 70)
    
    comparison_results = {}
    
    for dataset_name, data in results_5_1.items():
        print(f"\n📋 ANALISANDO: {dataset_name.upper()}")
        print("-" * 40)
        
        X_test = data['X_test']
        y_test = data['y_test']
        model = data['model']
        feature_names = data['feature_names']
        sample_indices = data['sample_indices']
        formal_attributions = data['formal_attributions']
        
        approx_methods = ApproximationMethods(model, feature_names)
        
        print("📊 Calculando importância por permutação...")
        perm_importance = approx_methods.permutation_importance(X_test, y_test)
        
        lime_attributions = []
        shap_attributions = []
        
        for idx in sample_indices:
            print(f"📈 Processando instância {idx}...")
            
            lime_attr = approx_methods.lime_attribution(X_test, idx)
            shap_attr = approx_methods.shap_approximation(X_test, idx)
            
            lime_attributions.append(lime_attr)
            shap_attributions.append(shap_attr)
        
        metrics = {}
