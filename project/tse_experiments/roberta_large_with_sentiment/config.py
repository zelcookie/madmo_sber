import os
import tokenizers

MAX_LEN = 192
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
EPOCHS = 7
SEED = 45
MAX_GRAD_NORM = 1.0
ROBERTA_PATH = '/home/mikhail/workspace/roberta-large'
MODEL_PATH = 'model.bin'
TRAINING_FILE = '../input/train.csv'
TOKENIZER = tokenizers.ByteLevelBPETokenizer(
    vocab_file=f"{ROBERTA_PATH}/vocab.json", 
    merges_file=f"{ROBERTA_PATH}/merges.txt", 
    lowercase=True,
    add_prefix_space=True
)