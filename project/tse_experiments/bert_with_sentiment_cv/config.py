import os
import tokenizers

MAX_LEN = 128
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 16
EPOCHS = 10
SEED = 43
BERT_PATH = '/home/mikhail/workspace/bert_base_uncased'
MODEL_PATH = 'model.bin'
TRAINING_FILE_WITHOUT_FOLDS = '../input/train.csv'
TRAINING_FILE = '../input/train_folds.csv'
VALID_FILE = '../input/valid.csv'
TOKENIZER = tokenizers.BertWordPieceTokenizer(
    os.path.join(BERT_PATH, 'vocab.txt'),
    lowercase=True
) 