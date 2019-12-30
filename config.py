"""Configuration for RNN base"""


class RNNConfig:

    model_name = 'AttLSTM'
    model_save_path = './ckpt/BiLSTMAtt'
    result_save_path = './result/BiLSTMAtt'

    learning_rate = 1.
    l2_scale = 1e-5
    embedding_size = 100
    hidden_size = 100
    dropout_keep_probs = [.7, .7, .5]
    epochs = 50
    batch_size = 10
    summary_step = 50
    lr_decay=.95
    epsilon=1e-8

    use_pre_weight = True
    train_pre_weight = True
    use_peepholes = True
    bilstm_merge_mode = 'add'    # or 'concat'
    att_activation = 'tanh'      # or 'relu'
    optimizer = 'adadelta'       # or 'adam'
    lr_schedule = 'constant'     # or 'standard'

    train_data_path = './data/korean_single_train.csv'
    dev_data_path = './data/korean_single_dev.csv'
    raw_data_path = './data/korean_single_turn_utterance_text.csv'
    tokenizer_path = './data/m32k.model'
    vocab_path = './data/m32k.vocab'
    w2v_path = './ckpt/wv/w2v_pretrained'
    num_classes = 7
    vocab_size = 32000
    max_len = 128
