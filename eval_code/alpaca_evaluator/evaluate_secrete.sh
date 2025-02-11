SIZE="70B"
REFERENCE="greedy"
OUTPUT_DIR="alpaca_outputs_${SIZE}"
API_KEY="YOUR_API_KEY" ##modify this for test
export OPENAI_BASE_URL=https://api.deepseek.com/v1
export OPENAI_API_KEY=${API_KEY}


CURRENT_DIR=$(pwd)
ALGOS=(
    "miro_t_1.0"
    "miro_t_1.5"
    "eta_t_1.0"
    "eta_t_1.5"
    "top_p_0.9_t_1.0"
    "top_p_0.9_t_1.5"
    "top_nsigma_1.0_t_1.0"
    "top_nsigma_1.0_t_1.5"
    "min_p_0.1_t_1.0"
    "min_p_0.1_t_1.5"
    "top_k_20_t_1.0"
    "top_k_20_t_1.5"
    "top_nsigma_1.0_t_3.0"
    "top_nsigma_1.0_t_10.0"
)

echo $CURRENT_DIR
set -e

for algo in ${ALGOS[@]}; do
    alpaca_eval evaluate --model_outputs ${CURRENT_DIR}/${OUTPUT_DIR}/${algo}.json \
        --reference_outputs ${CURRENT_DIR}/${OUTPUT_DIR}/${REFERENCE}.json \
        --annotators_config ${CURRENT_DIR}/alpaca_evaluator/deepseek.yaml \
        --output_path ${CURRENT_DIR}/alpaca_evaluator/results/ \
        --precomputed_leaderboard ${CURRENT_DIR}/alpaca_evaluator/results/leaderboard.csv
done

exit 0