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
       
        
        metrics = {}
        for i, idx in enumerate(sample_indices):
            formal_attr = formal_attributions[i]
            lime_attr = lime_attributions[i]
            shap_attr = shap_attributions[i]
            
            instance_metrics = {
                'correlations': calculate_correlations(formal_attr, lime_attr, shap_attr, perm_importance),
                'ranking_metrics': calculate_ranking_metrics(formal_attr, lime_attr, shap_attr, perm_importance)
            }
            metrics[idx] = instance_metrics
        
        comparison_results[dataset_name] = {
            'permutation_importance': perm_importance,
            'lime_attributions': lime_attributions,
            'shap_attributions': shap_attributions,
            'formal_attributions': formal_attributions,
            'sample_indices': sample_indices,
            'feature_names': feature_names,
            'metrics': metrics
        }
        
        print(f"✅ {len(sample_indices)} instâncias processadas")
        
        print("\n📊 RESUMO DAS CORRELAÇÕES:")
        for idx in sample_indices:
            corrs = metrics[idx]['correlations']
            print(f"   Instância {idx}:")
            print(f"     FFA vs LIME: {corrs['ffa_vs_lime']:.3f}")
            print(f"     FFA vs SHAP: {corrs['ffa_vs_shap']:.3f}")
            print(f"     LIME vs SHAP: {corrs['lime_vs_shap']:.3f}")
    
    print("\n" + "=" * 70)
    print("✅ SEÇÃO 5.2 CONCLUÍDA")
    print("=" * 70)
    
    return comparison_results