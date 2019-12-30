import os
import pandas as pd
import tensorflow as tf

from tokenizer import SentencePieceTokenizer
from utils import get_preweight, save_config
from model import AttentionRNN
from hparams import TrainingHparams
from generator import BatchGenerator


def save(config, results):
    result_df = pd.DataFrame(results)
    result_df.to_csv(os.path.join(config.result_save_path, 'train_val_result.csv'),
                     index=False)


def train(config, model, tokenizer):
    columns = ['epoch', 'trn_cost', 'trn_acc', 'val_cost', 'val_acc']
    results = {key: [] for key in columns}
    train_generator = BatchGenerator('train', config, tokenizer)
    dev_generator = BatchGenerator('dev', config, tokenizer)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, config.epochs+1):

            epoch_train_results = model.train(sess, epoch, config, train_generator)
            epoch_val_results = model.validate(sess, epoch, config, dev_generator)
            epoch_results = [epoch] + epoch_train_results + epoch_val_results

            for key, value in zip(results.keys(), epoch_results):
                results[key].append(value)

            model_name = str(epoch) + '-' + str(epoch_val_results[1])
            model.save(sess, config.model_save_path, model_name)
    return results


def get_tokenizer(config):
    """
    get Sentence Piece Tokenizer and pretrained word vectors if exists.
    :return: the tokenizer and pretrained weight in numpy array
    """
    tokenizer = SentencePieceTokenizer(config.tokenizer_path, config.vocab_path)
    if tokenizer.sp is None:
        print("creating tokenizer...")
        tokenizer.create_model(config.raw_data_path, 'm32k')
        print("Done.")

    if config.use_pre_weight:
        pre_weight = get_preweight(config.w2v_path, tokenizer)
    else:
        pre_weight = None
    return tokenizer, pre_weight


def main(config):
    tokenizer, pre_weight = get_tokenizer(config)
    model = AttentionRNN(config, pre_weight)

    results = train(config, model, tokenizer)
    print("training ended.")

    save(config, results)
    print("results saved.")


if __name__ == "__main__":
    args = TrainingHparams().parser.parse_args()
    save_config(args)
    main(args)
