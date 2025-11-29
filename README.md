# Formal Feature Attribution (FFA) — README

## Tema

Este repositório contém a implementação utilizada no relatório CoRR. O objetivo é gerar atribuição de recursos (*feature attribution*) exata e aproximada em **Árvores Impulsionadas por Gradiente (BTs)** com base na enumeração de explicações formais, aplicando o aparato da **IA Explicável Formal (XAI)**. A **Formal Feature Attribution (FFA)** é considerada vantajosa em relação aos métodos existentes, tanto formais quanto não formais.

---

## Definições rápidas

* **Formal Feature Attribution (FFA)**: Um método para gerar atribuição de recursos exata e aproximada em árvores impulsionadas por gradiente (BTs) com base na enumeração de explicações formais, aplicando o aparato da IA Explicável Formal (XAI). Argumenta-se que o FFA é vantajoso em relação aos métodos existentes, tanto formais quanto não formais.
* **Abductive Explanations (AXp's)**: Explicações que podem ser enumeradas como "Dual Explanations" para as árvores impulsionadas.

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
### 🛠️ Fluxo de Uso (Tutorial Básico)

O uso de exemplo está em `src/example.ipynb`. O fluxo padrão envolve 3 etapas principais.

#### 1. Preparar o Conjunto de Dados (`-p`)

O `FFA` trabalha com datasets em formato CSV. É necessário um arquivo `.catcol` listando os índices das colunas categóricas.

```bash
# Exemplo (para um arquivo 'dataset.csv' e um novo nome 'somename')
$ python explain.py -p --pfiles dataset.csv,somename somepath/
# Exemplo real
$ python explain.py -p --pfiles compas_train_data.csv,compas_train_data ../datasets/tabular/train/compas/
```

### 2. Treinar um Modelo Gradient Boosted Tree (`-c`)

Um modelo de árvore impulsionada por gradiente (BT) é requerido antes de gerar um *decision set*.

O valor do parâmetro `--testsplit` varia de `0.0` a `1.0`. Neste comando de exemplo, o dataset fornecido é dividido em 100% para treino e 0% para teste (`--testsplit 0`). O modelo gerado é salvo no caminho de saída especificado (`./btmodels/compas/`).

```bash
# Exemplo (25 árvores por classe, profundidade máxima 3)
$ python ./explain.py -o ./btmodels/compas/ -c --testsplit 0 -t -n 25 -d 3 ../datasets/tabular/train/compas/compas_train_data.csv
```
Neste exemplo, o modelo é salvo em um arquivo com nome similar a: `./btmodels/compas/compas_train_data/compas_train_data_nbestim_25_maxdepth_3_testsplit_0.0.mod.pkl.`

### 3. Enumerar Explicações Abductivas (AXp's) como Dual Explanations (`-e`)

Use este comando para enumerar explicações abdutivas ou contrastivas (AXp's) para BTs.

O parâmetro `--cut` é **opcional**. Seu valor indica o índice da instância específica para a qual se deseja enumerar explicações. Por padrão, todas as instâncias no dataset são consideradas. `<dataset.csv>` e `<model.pkl>` especificam o dataset de teste e o modelo BT treinado, respectivamente.

```bash
# Comando geral
$ python -u ./explain.py -e mx --am1 -E -T 1 -z -vvv -c --xtype <string> -R lin --sort abs --explain_ formal --xnum all -M --cut <int> --explains <dataset.csv> <model.pkl>

# Exemplo (para a instância de índice 5 do dataset compas)
$ python -u ./explain.py -e mx --am1 -E -T 1 -z -vvv -c --xtype con -R lin --sort abs --explain_ formal --xnum all -M --cut 5 --explains ../datasets/tabular/test/compas/compas_test_data.csv ./btmodels/compas/compas_train_data/compas_train_data_nbestim_25_maxdepth_3_testsplit_0.0.mod.pkl
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

## Instalação <a name="instl"></a>

### Método 1: Usando Conda (Recomendado)

```bash
# Clone o repositório
git clone [https://github.com/your-username/formal-feature-attribution.git](https://github.com/your-username/formal-feature-attribution.git)
cd formal-feature-attribution

# Crie e ative o ambiente conda
conda env create -f environment.yml
conda activate formal-feature-attribution
```
### Método 2: Usando Pip e Venv

```bash
# Clone o repositório
git clone [https://github.com/your-username/formal-feature-attribution.git](https://github.com/your-username/formal-feature-attribution.git)
cd formal-feature-attribution

# Crie um ambiente virtual
python -m venv venv

# Ative o ambiente virtual
# No Linux/Mac:
source venv/bin/activate
# No Windows:
venv\Scripts\activate

# Instale as dependências
pip install -r requirements.txt
```
### Método 3: Instalação para Desenvolvimento

```bash
# Clone e instale no modo de desenvolvimento
git clone [https://github.com/your-username/formal-feature-attribution.git](https://github.com/your-username/formal-feature-attribution.git)
cd formal-feature-attribution

# Usando conda
conda env create -f environment.yml
conda activate formal-feature-attribution

# Ou usando pip
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate no Windows
pip install -r requirements.txt

# Instale o pacote no modo de desenvolvimento
pip install -e .
```

### Verificação da Instalação

```bash
# Verifique se a instalação foi bem-sucedida
python -c "import src.formal_ffa; print('✅ FFA importado com sucesso!')"
python -c "import xgboost; import lime; print('✅ Dependências carregadas!')"

# Execute testes básicos
python -m pytest tests/ -v
```
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

## ⚠️ Soluções para Problemas Comuns

| Problema | Solução Linux/Mac | Solução Windows |
| :--- | :--- | :--- |
| **LIME com erro** | `$ sudo apt-get install python3-dev` (Ubuntu/Debian) ou `$ brew install python3` (Mac). | Garantir que o **Visual Studio Build Tools** está instalado. |
| **XGBoost com erro** | Instalar alternativamente: `pip install xgboost --upgrade` ou `conda install -c conda-forge xgboost`. | O mesmo que Linux/Mac. |
| **Conflito de versões** | Recriar o ambiente: `conda env remove -n formal-feature-attribution` e recriar com `conda env create -f environment.yml`. | O mesmo que Linux/Mac. |
