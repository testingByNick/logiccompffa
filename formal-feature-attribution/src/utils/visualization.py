import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, Any
import seaborn as sns


def plot_attribution_comparison(results: Dict[str, Any], dataset_name: str, 
                               instance_idx: int, save_path: str = None):
    """
    Plota comparação entre métodos de atribuição para uma instância específica
    
    Args:
        results: Resultados dos experimentos
        dataset_name: Nome do dataset
        instance_idx: Índice da instância
        save_path: Caminho para salvar o gráfico
    """
    plt.style.use('default')
    sns.set_palette("husl")
    
    feature_names = results['feature_names']
    formal_attr = results['formal_attributions'][instance_idx]
    lime_attr = results['lime_attributions'][instance_idx]
    shap_attr = results['shap_attributions'][instance_idx]
    perm_attr = results['permutation_importance']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Dataset: {dataset_name} - Instância {instance_idx}', 
                fontsize=16, fontweight='bold')
    
    axes[0, 0].barh(feature_names, formal_attr, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('FFA FORMAL', fontweight='bold')
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    axes[0, 1].barh(feature_names, lime_attr, color='lightcoral', alpha=0.7)
    axes[0, 1].set_title('LIME', fontweight='bold')
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].grid(axis='x', alpha=0.3)
    
    axes[1, 0].barh(feature_names, shap_attr, color='lightgreen', alpha=0.7)
    axes[1, 0].set_title('SHAP', fontweight='bold')
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    axes[1, 1].barh(feature_names, perm_attr, color='gold', alpha=0.7)
    axes[1, 1].set_title('PERMUTATION', fontweight='bold')
    axes[1, 1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Gráfico salvo em: {save_path}")
    
    plt.show()


def generate_final_report(results_5_1: Dict[str, Any], results_5_2: Dict[str, Any], 
                         output_dir: str = 'data/results/'):
    """
    Gera relatório final com análise completa dos resultados
    
    Args:
        results_5_1: Resultados da seção 5.1
        results_5_2: Resultados da seção 5.2
        output_dir: Diretório de saída
    """
    print("\n" + "=" * 70)
    print("📊 RELATÓRIO FINAL - ANÁLISE DOS RESULTADOS")
    print("=" * 70)
    
   