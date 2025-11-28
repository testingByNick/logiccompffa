import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import sys
import os

# Adiciona o caminho para importar módulos locais
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from formal_ffa import HeuristicFFA, FormalFFA


def run_section_5_1():
    """
    Executa experimentos da Seção 5.1: Datasets Sintéticos
    
    Returns:
        dict: Resultados dos experimentos
    """
    print("=" * 70)
    print("🎯 SEÇÃO 5.1: EXPERIMENTOS COM DATASETS SINTÉTICOS")
    print("=" * 70)
    
    results = {}
    
    # Configurações dos datasets conforme artigo
    dataset_configs = {
        'linear_separable': {
            'n_informative': 4,
            'n_redundant': 1, 
            'n_clusters_per_class': 1,
            'class_sep': 1.5
        },
        'non_linear': {
            'n_informative': 6,
            'n_redundant': 0,
            'n_clusters_per_class': 2,
            'class_sep': 0.8
        }
    }
    
    for dataset_name, config in dataset_configs.items():
        print(f"\n📊 DATASET: {dataset_name.upper()}")
        print("-" * 50)
        
        # Gera dataset sintético
        X, y = make_classification(
            n_samples=400,
            n_features=6,
            n_informative=config['n_informative'],
            n_redundant=config['n_redundant'],
            n_clusters_per_class=config['n_clusters_per_class'],
            class_sep=config['class_sep'],
            random_state=42,
            flip_y=0.05  # Pequeno ruído
        )
        
        # Normaliza para [0, 1]
        X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)
        
        # Split treino/teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Modelo XGBoost conforme artigo
        model = XGBClassifier(
            n_estimators=25,
            max_depth=3,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
        
        model.fit(X_train, y_train)
        # Avaliação do modelo
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        print(f"✅ Modelo treinado")
        print(f"   Acurácia - Treino: {train_acc:.3f}, Teste: {test_acc:.3f}")
        
        # Prepara para FFA
        feature_names = [f'F{i}' for i in range(X.shape[1])]
        feature_bounds = [(0, 1)] * X.shape[1]
        
        # Inicializa métodos FFA
        heuristic_ffa = HeuristicFFA(model, feature_names, X.shape[1])
        formal_ffa = FormalFFA(model, feature_bounds)
        
        # Seleciona instâncias para análise
        sample_indices = np.random.choice(len(X_test), size=2, replace=False)
        heuristic_attributions = []
        formal_attributions = []
        
        for idx in sample_indices:
            instance = X_test[idx]
            true_class = y_test[idx]
            
            # Calcula atribuições
            heuristic_attr = heuristic_ffa.compute(instance, true_class)
            formal_attr = formal_ffa.compute_ffa(instance, true_class)
            
            heuristic_attributions.append(heuristic_attr)
            formal_attributions.append(formal_attr)
            
            # Mostra feature mais importante
            top_heuristic = feature_names[np.argmax(heuristic_attr)]
            top_formal = feature_names[np.argmax(formal_attr)]
            
            print(f"📍 Instância {idx}:")
            print(f"   FFA Heurístico → {top_heuristic} ({heuristic_attr.max():.3f})")
            print(f"   FFA Formal → {top_formal} ({formal_attr.max():.3f})")
        
        # Armazena resultados
        results[dataset_name] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'model': model,
            'feature_names': feature_names,
            'feature_bounds': feature_bounds,
            'heuristic_attributions': heuristic_attributions,
            'formal_attributions': formal_attributions,
            'sample_indices': sample_indices,
            'performance': {'train_accuracy': train_acc, 'test_accuracy': test_acc}
        }
    
    print("\n" + "=" * 70)
    print("✅ SEÇÃO 5.1 CONCLUÍDA")
    print("=" * 70)
    
    return results