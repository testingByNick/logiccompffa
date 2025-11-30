# Reprodução Experimental: Atribuição Formal de Características

## Contexto da Pesquisa

Reprodução sistemática dos experimentos apresentados no artigo "On Formal Feature Attribution and Its Approximation", focando na implementação e validação de métodos formais de atribuição de características em modelos de machine learning.

## Conceitos Centrais

### Formal Feature Attribution (FFA) 

Este repositório contém a implementação utilizada no relatório CoRR. O objetivo é gerar atribuição de recursos (*feature attribution*) exata e aproximada em **Árvores Impulsionadas por Gradiente (BTs)** com base na enumeração de explicações formais, aplicando o aparato da **IA Explicável Formal (XAI)**. A **Formal Feature Attribution (FFA)** é considerada vantajosa em relação aos métodos existentes, tanto formais quanto não formais.

---

### Explicações Abdutivas (AXp's)
Conjuntos mínimos de características que, quando fixadas, garantem determinada predição para qualquer combinação das demais features. Representam o núcleo da abordagem formal.

### Métodos de Aproximação
Técnicas heurísticas incluindo LIME (explicações locais), SHAP (valores de Shapley) e importância por permutação, utilizadas como baseline para comparação.

## Objetivos da Reprodução

- Implementar o cálculo formal de atribuição conforme definição matemática do artigo
- Validar experimentalmente a superioridade de métodos formais sobre abordagens heurísticas
- Reproduzir resultados das seções experimentais 5.1 e 5.2 do artigo original
- Fornecer implementação de referência para pesquisas futuras

---

## Definições rápidas

* **Formal Feature Attribution (FFA)**: Um método para gerar atribuição de recursos exata e aproximada em árvores impulsionadas por gradiente (BTs) com base na enumeração de explicações formais, aplicando o aparato da IA Explicável Formal (XAI). Argumenta-se que o FFA é vantajoso em relação aos métodos existentes, tanto formais quanto não formais.

---

## Metodologia de Uso e Implementação

O repositório contém a implementação utilizada no relatório CoRR. Antes de usar, é necessário extrair os conjuntos de dados contidos em `datasets.tar.xz`. Para extrair, use:


$ tar -xvf datasets.tar.xz
### 📦 Requisitos e Instalação
A implementação é feita em scripts Python (versão 3.8.5 usada nos experimentos).

### Pacotes Necessários: Os requisitos estão listados em requirements.txt. Instale-os via pip install -r requirements.txt.

### Extração de Dados:

```bash

$ tar -xvf datasets.tar.xz
```

#### Métodos de Instalação (Recomendado: Conda)

| Método | Passos Chave |
| :--- | :--- |
| **Conda** (Recomendado) | Clonar o repositório, `conda env create -f environment.yml`, `conda activate formal-feature-attribution`. |
| **Pip e Venv** | Clonar, criar `venv`, ativar (`source venv/bin/activate` ou `venv\Scripts\activate`), `pip install -r requirements.txt`. |
| **Desenvolvimento** | Instalar dependências, `pip install -e .` (para modo de edição). |

---
### Verificação da Instalação

```bash

# Execute testes básicos
python -m pytest tests/ -v

```
---

## Para Desenvolvedores 

```bash
# Instalação com dependências de desenvolvimento
pip install -r requirements.txt
pip install -e .[dev]  # Se houver extras de desenvolvimento

# Configure os hooks pre-commit (opcional)
pre-commit install
```


### Principais Dependências

O projeto utiliza amplamente os seguintes frameworks e *solver*:

* **XGBoost**: Chen & Guestrin (2016) - Modelos ensemble.
* **LIME**: Ribeiro et al. (2016) - Explicações locais.
* **SHAP**: Lundberg & Lee (2017) - Valores de Shapley.
* **Z3**: Microsoft Research - Solver SAT/SMT.

---

### 🛠️ Fluxo de Uso (Tutorial Básico)

#### 1. Reprodução de Experimentos (Script Principal)

Para reproduzir os resultados das seções 5.1 e 5.2 do artigo, utilize o script `run_experiments.py`. Este script gerencia a execução dos testes e a geração dos relatórios finais.

**Argumentos Disponíveis:**
* `--section`: Escolha quais experimentos rodar (`5.1`, `5.2` ou `all`). O padrão é `all`.
* `--output-dir`: Diretório onde os resultados e relatórios serão salvos (padrão: `data/results/`).

**Exemplo de Execução:**

```bash
# Executar todos os experimentos e salvar em pasta customizada
python run_experiments.py --section all --output-dir ./meus_resultados
```


---

## 🎯 Reprodução Experimental

* Devido à **aleatoriedade** usada no processo de amostragem em **LIME** e **SHAP**, é improvável que os resultados experimentais relatados na submissão possam ser *completamente* reproduzidos.
* Resultados **semelhantes** podem ser obtidos com o seguinte script:
    ```bash
    $ cd ./src/ & ./experiment/repro_exp.sh
    ```
* A execução dos experimentos levará algum tempo, pois o número total de datasets e instâncias consideradas é grande.

---

## 📚 Referências

### Artigo Principal

```bibtex
@article{yu2023formal,
  title={On Formal Feature Attribution and Its Approximation},
  author={Yu, Jinqiang and Ignatiev, Alexey and Stuckey, Peter J.},
  journal={arXiv preprint arXiv:230X.XXXXX},
  year={2023}
}
```

---

## ⚠️ Soluções para Problemas Comuns

| Problema | Solução Linux/Mac | Solução Windows |
| :--- | :--- | :--- |
| **LIME com erro** | `$ sudo apt-get install python3-dev` (Ubuntu/Debian) ou `$ brew install python3` (Mac). | Garantir que o **Visual Studio Build Tools** está instalado. |
| **XGBoost com erro** | Instalar alternativamente: `pip install xgboost --upgrade` ou `conda install -c conda-forge xgboost`. | O mesmo que Linux/Mac. |
| **Conflito de versões** | Recriar o ambiente: `conda env remove -n formal-feature-attribution` e recriar com `conda env create -f environment.yml`. | O mesmo que Linux/Mac. |

---

## Contribuições da Reprodução

### Para a Comunidade Científica

- Implementação de referência do método FMA formal
- Validação independente dos resultados do artigo original
- Base código aberto para extensões e pesquisas futuras
- Documentação detalhada do processo experimental

### Para Prática em Explainable AI

- Demonstração prática das vantagens de métodos formais
- Identificação de cenários onde métodos heurísticos falham
- Framework para avaliação crítica de explicações de modelos
- Guia para implementação de verificações formais

## Limitações e Desenvolvimentos Futuros

### Restrições Atuais

- Complexidade computacional em verificações formais completas
- Escala limitada comparada a alguns experimentos do artigo
- Dependência de amostragem para casos de grande dimensionalidade

### Direções Futuras

- Implementação de algoritmos otimizados para enumeração de AXp's
- Expansão para datasets de maior escala e complexidade
- Integração com outros paradigmas de modelos de ML
- Desenvolvimento de técnicas híbridas formais-heurísticas

## Conclusão

Esta reprodução experimental estabelece uma base sólida para compreensão e aplicação de métodos formais de atribuição de características, validando suas vantagens teóricas através de implementação prática e análise sistemática. Os resultados reforçam a importância de abordagens com garantias formais em aplicações críticas de machine learning explicável.

