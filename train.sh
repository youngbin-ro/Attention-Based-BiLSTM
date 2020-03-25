NAME="BiLSTMAtt_FinalBaseline"
MODEL_PATH="./ckpt/${NAME}"
RESULT_PATH="./result/${NAME}"

mkdir -p ${MODEL_PATH}
mkdir -p ${RESULT_PATH}

python train.py \
  --model_name=${NAME} \
  --model_save_path=${MODEL_PATH} \
  --result_save_path=${RESULT_PATH}
