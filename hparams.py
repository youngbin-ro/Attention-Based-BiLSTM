import argparse
from config import RNNConfig as BaseConfig
from utils import str2bool


class TrainingHparams:
    parser = argparse.ArgumentParser(description='hyperparameters for training')

    # paths and files
    parser.add_argument("--model_name", type=str, default=BaseConfig.model_name,
                        help="name of the model to train")
    parser.add_argument("--model_save_path", type=str, default=BaseConfig.model_save_path,
                        help="directory to save the model checkpoints")
    parser.add_argument("--result_save_path", type=str, default=BaseConfig.result_save_path,
                        help="directory to save the training results")

    # numerical parameters
    parser.add_argument("--learning_rate", type=float, default=BaseConfig.learning_rate,
                        help="learning rate for training")
    parser.add_argument("--l2_scale", type=float, default=BaseConfig.l2_scale,
                        help="scale of l2 regularization (weight decay)")
    parser.add_argument("--embedding_size", type=int, default=BaseConfig.embedding_size,
                        help="embedding dimension size of word vectors")
    parser.add_argument("--hidden_size", type=int, default=BaseConfig.hidden_size,
                        help="hidden node dimension of LSTM model")
    parser.add_argument("--emb_keep_prob", type=float, default=BaseConfig.dropout_keep_probs[0],
                        help="keep probability on dropout in embedding layer")
    parser.add_argument("--rnn_keep_prob", type=float, default=BaseConfig.dropout_keep_probs[1],
                        help="keep probability on dropout in bidirectional lstm layer")
    parser.add_argument("--att_keep_prob", type=float, default=BaseConfig.dropout_keep_probs[2],
                        help="keep probability on dropout in attention layer")
    parser.add_argument("--epochs", type=int, default=BaseConfig.epochs,
                        help="epochs for training")
    parser.add_argument("--batch_size", type=int, default=BaseConfig.batch_size,
                        help="mini batch size of training data")
    parser.add_argument("--summary_step", type=int, default=BaseConfig.summary_step,
                        help="step interval to display the training results")
    parser.add_argument("--lr_decay", type=float, default=BaseConfig.lr_decay,
                        help="learning rate decay of Adadelta optimizer")
    parser.add_argument("--epsilon", type=float, default=BaseConfig.epsilon,
                        help="epsilon parameter of Adadelta optimizer")

    # bool anf string type parameters
    parser.add_argument("--use_pre_weight", type=str2bool, nargs='?', const=True,
                        default=BaseConfig.use_pre_weight,
                        help="whether to use pretrained word vectors")
    parser.add_argument("--train_pre_weight", type=str2bool, nargs='?', const=True,
                        default=BaseConfig.train_pre_weight,
                        help="whether to make pretrained word vectors trainable in training phase")
    parser.add_argument("--use_peepholes", type=str2bool, nargs='?', const=True,
                        default=BaseConfig.use_peepholes,
                        help="whether to use peephole connections in LSTM cells")
    parser.add_argument("--bilstm_merge_mode", type=str, default=BaseConfig.bilstm_merge_mode,
                        help="merge mode of forward and backward output in LSTM. 'add' or 'concat' supported")
    parser.add_argument("--att_activation", type=str, default=BaseConfig.att_activation,
                        help="activation function of attention layer. 'tanh' or 'relu' supported")
    parser.add_argument("--optimizer", type=str, default=BaseConfig.optimizer,
                        help="optimizer of the model. 'adadelta' or 'adam' supported")
    parser.add_argument("--lr_schedule", type=str, default=BaseConfig.lr_schedule,
                        help="learning rate schedule of training. 'standard' or 'constant' supported")

    # base configuration (rarely change)
    parser.add_argument("--train_data_path", type=str, default=BaseConfig.train_data_path,
                        help="csv file of training dataset")
    parser.add_argument("--dev_data_path", type=str, default=BaseConfig.dev_data_path,
                        help="csv file of dev(validation) dataset")
    parser.add_argument("--raw_data_path", type=str, default=BaseConfig.raw_data_path,
                        help="directory of total raw dataset")
    parser.add_argument("--tokenizer_path", type=str, default=BaseConfig.tokenizer_path,
                        help="directory of trained tokenizer")
    parser.add_argument("--vocab_path", type=str, default=BaseConfig.vocab_path,
                        help="directory of vocabulary set from tokenizer")
    parser.add_argument("--w2v_path", type=str, default=BaseConfig.w2v_path,
                        help="directory of pretrained word vectors")
    parser.add_argument("--num_classes", type=int, default=BaseConfig.num_classes,
                        help="number of classes to classify with model")
    parser.add_argument("--vocab_size", type=int, default=BaseConfig.vocab_size,
                        help="number of vocabs in vocab file")
    parser.add_argument("--max_len", type=int, default=BaseConfig.max_len,
                        help="maximum sequence length of the total dataset")
