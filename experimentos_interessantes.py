#Método de aproximação (algoritmo) — XPENUM / estratégia prática

#Quem apresenta: Membro C
#O que cobrir (slides):

#Por que é preciso aproximar (FFA exata nem sempre viável).

#Explicar a estratégia principal do artigo: usar enumeração contínua tipo MARCO, mas mirando CXp para rapidamente obter muitas explicações duais e assim coletar AXp “de graça” — ideia do Algoritmo 1 (XPENUM). Mostre pseudo-código/fluxo.

#Como a aproximação converge (monotonicidade da FFA aproximada conforme aumenta o tempo/numero de explicações).

#Implementação prática (protótipo em Python, repositório do código disponível).

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$REPO_ROOT/src"
OUT_DIR="$SRC_DIR/experiment/results_tabular"
mkdir -p "$OUT_DIR/logs" "$OUT_DIR/metrics" "$OUT_DIR/models"

# Parâmetros (ajuste aqui)
N_ESTIMATORS=25
MAX_DEPTH=3
TIME_BUDGETS=(10 30 120 600)   # segundos para FFA approximate runs (ex.: FFA10, FFA30...)
DATASETS=(compas adult pima qtw) # lista curta de exemplo - ajuste para os datasets que existem em datasets/tabular
# Nota: substitua 'qtw' por nomes reais dos datasets disponíveis no seu datasets/tabular

# Função: treina e executa pipeline por dataset
run_dataset() {
  local ds="$1"
  echo "==== Dataset: $ds ===="
  ds_train="../datasets/tabular/train/${ds}/${ds}_train_data.csv"
  ds_test="../datasets/tabular/test/${ds}/${ds}_test_data.csv"

  if [ ! -f "$SRC_DIR/$ds_train" ] && [ ! -f "$REPO_ROOT/$ds_train" ]; then
    echo "Aviso: dataset $ds não encontrado em $REPO_ROOT/datasets/tabular/... — pulando."
    return
  fi

  # Treina modelo BT
  model_out="$OUT_DIR/models/${ds}_nbestim_${N_ESTIMATORS}_maxdepth_${MAX_DEPTH}.mod.pkl"
  if [ ! -f "$model_out" ]; then
    echo "Treinando BT para $ds..."
    python "$SRC_DIR/explain.py" -o "$OUT_DIR/models/${ds}/" -c --testsplit 0 -t -n ${N_ESTIMATORS} -d ${MAX_DEPTH} "$REPO_ROOT/datasets/tabular/train/${ds}/${ds}_train_data.csv" \
      2>&1 | tee "$OUT_DIR/logs/train_${ds}.log"
    # o explain.py salva modelo em ./btmodels/* por padrão; mova para nosso model_out padrão se necessário
    # procure arquivo .mod.pkl gerado e copie para model_out
    find . -maxdepth 4 -type f -name "${ds}_train_data_nbestim_*_maxdepth_${MAX_DEPTH}_testsplit_0.0.mod.pkl" -print -exec cp {} "$model_out" \; || true
  else
    echo "Modelo já existe: $model_out"
  fi

  # Executar LIME e SHAP (assumimos utilitários existentes no src/experiment ou explain.py)
  # aqui chamamos scripts de baseline prontos (se existirem) - caso contrário, usamos o notebook example
  echo "Rodando LIME e SHAP para $ds..."
  python "$SRC_DIR/experiment/run_baselines_tabular.py" --dataset "$ds" --model "$model_out" --outdir "$OUT_DIR" \
    2>&1 | tee "$OUT_DIR/logs/baselines_${ds}.log"

  # Executar aproximações FFA para cada budget (chamando explain.py com tempo limite)
  for budget in "${TIME_BUDGETS[@]}"; do
    echo "Rodando FFA (budget=${budget}s) para $ds..."
    # interpretamos que explain.py suporta timeout externo; usamos timeout do shell
    timeout "${budget}s" python "$SRC_DIR/explain.py" -e mx --am1 -E -T 1 -z -vvv -c --xtype con -R lin --sort abs \
      --explain_formal --xnum all -M --cut all --explains "$REPO_ROOT/datasets/tabular/test/${ds}/${ds}_test_data.csv" "$model_out" \
      2>&1 | tee "$OUT_DIR/logs/ffa_${ds}_${budget}s.log" || echo "timeout ou erro (esperado se limitou)"
    # Extraia métricas do log com um utilitário (run_eval_tabular.py)
    python "$SRC_DIR/experiment/run_eval_tabular.py" --log "$OUT_DIR/logs/ffa_${ds}_${budget}s.log" \
      --baseline "$OUT_DIR/baselines_${ds}.json" --out "$OUT_DIR/metrics/${ds}_ffa_${budget}s.metrics.json"
  done

  echo "Fim dataset $ds"
}

# Execução principal
for ds in "${DATASETS[@]}"; do
  run_dataset "$ds"
done

echo "Todos os datasets processados. Resultados em $OUT_DIR"






#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$REPO_ROOT/src"
OUT_DIR="$SRC_DIR/experiment/results_mnist"
mkdir -p "$OUT_DIR/logs" "$OUT_DIR/heatmaps" "$OUT_DIR/metrics" "$OUT_DIR/models"

N_ESTIMATORS=25
MAX_DEPTH=3
TIME_BUDGETS=(10 30 120 600 1200 3600)
DATASETS=("10x10-mnist-1vs3" "10x10-mnist-1vs7" "28x28-mnist-1vs3" "28x28-mnist-1vs7")

# Função principal
run_mnist() {
  local ds="$1"
  echo "=== MNIST dataset $ds ==="
  model_out="$OUT_DIR/models/${ds}_nbestim_${N_ESTIMATORS}_maxdepth_${MAX_DEPTH}.mod.pkl"
  # Treine se necessário (reutiliza explain.py training mode)
  if [ ! -f "$model_out" ]; then
    echo "Treinando modelo BT para $ds..."
    python "$SRC_DIR/explain.py" -o "$OUT_DIR/models/${ds}/" -c --testsplit 0 -t -n ${N_ESTIMATORS} -d ${MAX_DEPTH} "$REPO_ROOT/datasets/image/${ds}/${ds}_train.csv" \
      2>&1 | tee "$OUT_DIR/logs/train_${ds}.log"
    find . -maxdepth 4 -type f -name "*${ds}*_nbestim_*_maxdepth_${MAX_DEPTH}_testsplit_0.0.mod.pkl" -exec cp {} "$model_out" \; || true
  fi

  # Run LIME/SHAP baseline
  python "$SRC_DIR/experiment/run_baselines_image.py" --dataset "$ds" --model "$model_out" --outdir "$OUT_DIR" \
    2>&1 | tee "$OUT_DIR/logs/baselines_${ds}.log"

  # Run FFA approximations for budgets
  for budget in "${TIME_BUDGETS[@]}"; do
    echo "FFA budget=${budget}s for $ds"
    timeout "${budget}s" python "$SRC_DIR/explain.py" -e mx --am1 -E -T 1 -z -vvv -c --xtype con -R lin --sort abs \
      --explain_formal --xnum all -M --cut all --explains "$REPO_ROOT/datasets/image/${ds}/${ds}_test.csv" "$model_out" \
      2>&1 | tee "$OUT_DIR/logs/ffa_${ds}_${budget}s.log" || echo "timeout (esperado)"
    # gerar heatmaps a partir dos logs (usando util run_eval_image.py)
    python "$SRC_DIR/experiment/run_eval_image.py" --log "$OUT_DIR/logs/ffa_${ds}_${budget}s.log" --outdir "$OUT_DIR/heatmaps" \
      --baseline "$OUT_DIR/baselines_${ds}.json" --metrics "$OUT_DIR/metrics/${ds}_ffa_${budget}s.metrics.json"
  done
  echo "Fim MNIST $ds"
}

for ds in "${DATASETS[@]}"; do
  run_mnist "$ds"
done

echo "MNIST experiments done. Results in $OUT_DIR"






#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$REPO_ROOT/src"
OUT_DIR="$SRC_DIR/experiment/results_jit"
mkdir -p "$OUT_DIR/logs" "$OUT_DIR/metrics" "$OUT_DIR/models"

DATASETS=(openstack qt)
N_ESTIMATORS=25
MAX_DEPTH=3

for ds in "${DATASETS[@]}"; do
  echo "=== JIT dataset: $ds ==="
  # train logistic/regression or load preprocessed train/test from repo
  model_out="$OUT_DIR/models/${ds}_logistic.mod.pkl"
  if [ ! -f "$model_out" ]; then
    echo "Treinando modelo de regressão logística para $ds..."
    python "$SRC_DIR/experiment/train_jit_model.py" --dataset "$REPO_ROOT/datasets/jit/${ds}.csv" --out "$model_out" \
      2>&1 | tee "$OUT_DIR/logs/train_jit_${ds}.log"
  fi

  # extrair FFA exato (o artigo afirma que para regressão logística é rápido)
  python "$SRC_DIR/explain.py" -e mx --am1 -E -T 1 -z -vvv -c --xtype con -R lin --sort abs \
    --explain_formal --xnum all -M --cut all --explains "$REPO_ROOT/datasets/jit/${ds}_test.csv" "$model_out" \
    2>&1 | tee "$OUT_DIR/logs/ffa_jit_${ds}.log"

  # gerar métricas comparativas com LIME/SHAP (baseline script)
  python "$SRC_DIR/experiment/run_eval_jit.py" --log "$OUT_DIR/logs/ffa_jit_${ds}.log" \
    --baseline "$OUT_DIR/baselines_jit_${ds}.json" --out "$OUT_DIR/metrics/${ds}_jit.metrics.json" \
    2>&1 | tee "$OUT_DIR/logs/eval_jit_${ds}.log"
done

echo "JIT experiments finished. Results in $OUT_DIR"

