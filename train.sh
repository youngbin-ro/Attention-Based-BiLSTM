NAME="BiLSTMAtt_PaperConfig3"
MODEL_PATH="./ckpt/${NAME}"
RESULT_PATH="./result/${NAME}"

mkdir -p ${MODEL_PATH}
mkdir -p ${RESULT_PATH}

python train.py --model_name=${NAME}\
                --model_save_path=${MODEL_PATH}\
                --result_save_path=${RESULT_PATH}\
                --epochs=100

NAME="BiLSTMAtt_PaperConfig4"
MODEL_PATH="./ckpt/${NAME}"
RESULT_PATH="./result/${NAME}"

mkdir -p ${MODEL_PATH}
mkdir -p ${RESULT_PATH}

python train.py --model_name=${NAME}\
                --model_save_path=${MODEL_PATH}\
                --result_save_path=${RESULT_PATH}\
                --lr_schedule="standard"\
                --lr_decay=.9\
                --epsilon=1e-6\
                --epochs=100
