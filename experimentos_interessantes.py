 Essas funções representam chamadas ao raciocínio formal (SAT / SMT).
import time
from collections import defaultdict
def model_predict(model, x):
    """
    Retorna a classe prevista pelo modelo κ para a instância x.
    """
    return model(x)

def is_contrastive_explanation(model, v, c, Y):
    """
    Verifica se Y é uma explicação contrastiva (CXp).

    Retorna True se existir uma instância x tal que:
    - Features fora de Y permanecem iguais a v
    - A predição muda (κ(x) != c)
    """
    # Aqui estaria uma consulta SAT/SMT
    # Simulação didática:
    return bool(hash(frozenset(Y)) % 2)

def extract_abductive_explanation(model, v, c, fixed_features):
    """
    Extrai uma explicação abdutiva mínima (AXp),
    dado o conjunto de features fixadas.
    """
    # Em prática: remoção iterativa com verificação formal
    # Aqui simulamos:
    return set(fixed_features)

# Algoritmo principal
def approximate_ffa(model, v, c, time_limit):
    """
    Aproxima a FFA usando enumeração progressiva de explicações.
    """
    start = time.time()

    AXp = []  # Conjunto de explicações abdutivas
    CXp = []  # Conjunto de explicações contrastivas

    features = set(v.keys())

    while time.time() - start < time_limit:
        # Gera um conjunto candidato de features (hitting set mínimo)
        Y = set(list(features)[:max(1, len(features) // 2)])

        if is_contrastive_explanation(model, v, c, Y):
            CXp.append(Y)
        else:
            X = extract_abductive_explanation(model, v, c, features - Y)
            AXp.append(X)

    return AXp

# Calculo de atribuição
def compute_ffa(AXp):
    """
    Calcula a atribuição formal aproximada das features.
    """
    feature_count = defaultdict(int)

    for explanation in AXp:
        for feature in explanation:
            feature_count[feature] += 1

    total = len(AXp)

    ffa = {
        feature: count / total
        for feature, count in feature_count.items()
    }

    return ffa

# Essas funções representam chamadas ao raciocínio formal (SAT / SMT).
def model_predict(model, x):
    """
    Retorna a classe prevista pelo modelo κ para a instância x.
    """
    return model(x)

def is_contrastive_explanation(model, v, c, Y):
    """
    Verifica se Y é uma explicação contrastiva (CXp).

    Retorna True se existir uma instância x tal que:
    - Features fora de Y permanecem iguais a v
    - A predição muda (κ(x) != c)
    """
    # Aqui estaria uma consulta SAT/SMT
    # Simulação didática:
    return bool(hash(frozenset(Y)) % 2)

def extract_abductive_explanation(model, v, c, fixed_features):
    """
    Extrai uma explicação abdutiva mínima (AXp),
    dado o conjunto de features fixadas.
    """
    # Em prática: remoção iterativa com verificação formal
    # Aqui simulamos:
    return set(fixed_features)

# Algoritmo principal
def approximate_ffa(model, v, c, time_limit):
    """
    Aproxima a FFA usando enumeração progressiva de explicações.
    """
    start = time.time()

    AXp = []  # Conjunto de explicações abdutivas
    CXp = []  # Conjunto de explicações contrastivas

    features = set(v.keys())

    while time.time() - start < time_limit:
        # Gera um conjunto candidato de features (hitting set mínimo)
        Y = set(list(features)[:max(1, len(features) // 2)])

        if is_contrastive_explanation(model, v, c, Y):
            CXp.append(Y)
        else:
            X = extract_abductive_explanation(model, v, c, features - Y)
            AXp.append(X)

    return AXp

# Calculo de atribuição
def compute_ffa(AXp):
    """
    Calcula a atribuição formal aproximada das features.
    """
    feature_count = defaultdict(int)

    for explanation in AXp:
        for feature in explanation:
            feature_count[feature] += 1

    total = len(AXp)

    ffa = {
        feature: count / total
        for feature, count in feature_count.items()
    }

    return ffa

# Exemplo
# Instância fictícia
v = {
    "education": "bachelor",
    "hours_per_week": 40,
    "status": "single"
}

# Modelo fictício
def model(x):
    return "<50k"

# Classe prevista
c = model(v)

AXp = approximate_ffa(
    model=model,
    v=v,
    c=c,
    time_limit=2  # segundos
)

ffa = compute_ffa(AXp)

print("Atribuição Formal Aproximada:")
for k, v_val in ffa.items(): # Renamed 'v' to 'v_val' to avoid conflict with instance 'v'
    print(f"{k}: {v_val:.2f}")
