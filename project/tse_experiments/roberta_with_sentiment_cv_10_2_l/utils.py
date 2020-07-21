import random
import torch
import os
import numpy as np
import config
from sklearn.model_selection import StratifiedKFold, train_test_split
import pandas as pd


def get_selected_text(text, start_idx, end_idx, offsets):
    selected_text = ""
    for ix in range(start_idx, end_idx + 1):
        selected_text += text[offsets[ix][0]: offsets[ix][1]]
        if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
            selected_text += " "
    return selected_text

def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def compute_jaccard_score(text, start_idx, end_idx, start_logits, end_logits, offsets):
    start_pred = np.argmax(start_logits)
    end_pred = np.argmax(end_logits)
    if start_pred > end_pred:
        pred = text
    else:
        pred = get_selected_text(text, start_pred, end_pred, offsets)
        
    true = get_selected_text(text, start_idx, end_idx, offsets)
    
    return jaccard(true, pred)


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def add_folds(with_valid=True):
    if with_valid:
        dfx = pd.read_csv(config.TRAINING_FILE_WITHOUT_FOLDS).dropna().reset_index(drop=True)
        df_train, df_valid = train_test_split(
            dfx,
            test_size=0.1,
            random_state=42,
            stratify=dfx.sentiment.values
        )

        df_train = df_train.reset_index(drop=True)
        df_valid = df_valid.reset_index(drop=True)
        skf = StratifiedKFold(n_splits=config.FOLDS, shuffle=True, random_state=config.SEED)
        df_train['kfold'] = -1
        for num, (_, inds) in enumerate(skf.split(df_train.text, df_train.sentiment)):
            df_train.iloc[inds, -1] = num
        df_train.to_csv(config.TRAINING_FILE, index=False)
        df_valid.to_csv(config.VALID_FILE, index=False)
    else:
        df_train = pd.read_csv(config.TRAINING_FILE_WITHOUT_FOLDS).dropna().reset_index(drop=True)
        skf = StratifiedKFold(n_splits=config.FOLDS, shuffle=True, random_state=config.SEED)
        df_train['kfold'] = -1
        for num, (_, inds) in enumerate(skf.split(df_train.text, df_train.sentiment)):
            df_train.iloc[inds, -1] = num
        df_train.to_csv(config.TRAINING_FILE, index=False)