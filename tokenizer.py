import sentencepiece as spm
import pandas as pd


class SentencePieceTokenizer:

    def __init__(self, model_path=None, vocab_path=None):
        """
        :param model_path: 학습된 모델 경로 (ex. ./m.model)
        :param vocab_path: 학습된 단어 경로 (ex. ./m.vocab)
        sentencepiece로 학습된 모델이 없을 경우 create_model Method를 이용해 학습 진행
        """
        self.sp = None
        self.word2idx = None

        if model_path is not None:
            self.load_model(model_path)

        if vocab_path is not None:
            self.load_vocab(vocab_path)

    def load_model(self, path):
        """학습된 sentencepiece tokenizer를 로드(m.model)"""

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(path)

    def load_vocab(self, path):
        """학습된 sentencepiece 단어를 로드(m.vocab)"""

        with open(path, encoding='utf-8') as f:
            Vo = [doc.strip().split("\t") for doc in f]

        self.word2idx = {w[0]: i for i, w in enumerate(Vo)}

    def tokenize(self, data, add_tokens=True, to_ids=False):
        """ tokenize data
        ex) 오늘도 평화로운 연구실
        -> ['<s>', '▁오늘도', '▁평화', '로운', '▁연구', '실', '</s>']
        :param data: list or Series of sentences
        :param add_tokens: <s>, </s>를 양 끝에 반환
        :param to_ids: wordpiece가 아닌 index로 tokenize
        :return: tokenized sentence
        """
        assert type(data) in [pd.Series, list, str], 'Series, list, str 인자만 입력 가능합니다.'
        if self.sp is None:
            raise ValueError("Load Sentencepiece model before tokenizing")

        if add_tokens:
            # 문장 양 끝에 <s> , </s> 추가
            self.sp.SetEncodeExtraOptions('bos:eos')

        def enc(x):
            return self.sp.EncodeAsIds(x) if to_ids else self.sp.EncodeAsPieces(x)

        if type(data) is str:
            return enc(data)

        return list(map(enc, data))

    def decode(self, encoded_sentence, from_id=False):
        """untokenize
        :param encoded_sentence: encoded sentence(list)
        :param is_id: wheter input sentence is encoded as indices
        ex) ['<s>', '▁오늘도', '▁평화', '로운', '▁연구', '실', '</s>']
        -> 오늘도 평화로운 연구실
        """
        assert type(encoded_sentence) is list, 'list 인자만 입력 가능합니다/'
        if self.sp is None:
            raise ValueError("Load Sentencepiece model before tokenizing")

        if from_id:
            return self.sp.decode_ids(encoded_sentence)

        return self.sp.decode_pieces(encoded_sentence)

    def create_model(self, data_path,
                     output_name,
                     vocab_size=32000,
                     pad_id=0,
                     method='unigram'):
        """
        sentencepiece를 학습시키고 model과 vocab을 로드
        :param data_path: 데이터 경로. list of sentences
        :param output_name: output file name
        :param vocab_size: predetermined vocabulary size
        :param method: Choose from unigram (default), bpe, char, or word
        :return: model과 vocab을 저장한 뒤 인스턴스 객체로 설정
        """
        spm.SentencePieceTrainer_Train('--input=' + data_path +
                                       ' --pad_id=' + str(pad_id) +
                                       ' --bos_id=1' +
                                       ' --eos_id=2' +
                                       ' --unk_id=3' +
                                       ' --model_prefix=' + output_name +
                                       ' --vocab_size=' + str(vocab_size) +
                                       ' --character_coverage=1.0' +
                                       ' --model_type=' + method)

        self.load_model(output_name + '.model')
        self.load_vocab(output_name + '.vocab')
