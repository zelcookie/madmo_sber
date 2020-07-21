import os
import tokenizers

MAX_LEN = 96
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 3
SEED = 42
FOLDS = 10
ROBERTA_PATH = '/home/mikhail/workspace/roberta-base'
MODEL_PATH = 'model.bin'
TRAINING_FILE_WITHOUT_FOLDS = '../input/train.csv'
TRAINING_FILE = '../input/train_folds.csv'
VALID_FILE = '../input/valid.csv'
TOKENIZER = tokenizers.ByteLevelBPETokenizer(
    vocab_file=f"{ROBERTA_PATH}/vocab.json", 
    merges_file=f"{ROBERTA_PATH}/merges.txt", 
    lowercase=True,
    add_prefix_space=True
)