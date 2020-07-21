import config
from dataset import TweetDataset, get_train_val_loaders
import engine
import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import utils 

from utils import seed_everything
from model import TweetModel
from sklearn import model_selection
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import transformers

def loss_fn(start_logits, end_logits, start_positions, end_positions):
    ce_loss = nn.CrossEntropyLoss()
    start_loss = ce_loss(start_logits, start_positions)
    end_loss = ce_loss(end_logits, end_positions)    
    total_loss = start_loss + end_loss
    return total_loss




def run(fold):
    dfx = pd.read_csv(config.TRAINING_FILE)

    df_train = dfx[dfx.kfold != fold].reset_index(drop=True)
    df_valid = dfx[dfx.kfold == fold].reset_index(drop=True)
    
    model_config = transformers.RobertaConfig.from_pretrained('/home/mikhail/workspace/roberta-base/')
    model_config.output_hidden_states = True
    model = TweetModel(model_config)
    optimizer = AdamW(model.parameters(), lr=3e-5, betas=(0.9, 0.999))
    criterion = loss_fn    
    dataloaders_dict = get_train_val_loaders(df_train, df_valid , config.TRAIN_BATCH_SIZE)

    engine.train_model(
        model, 
        dataloaders_dict,
        criterion, 
        optimizer, 
        config.EPOCHS,
        f'roberta_fold{fold}.pth')


if __name__=='__main__':
    seed_everything(config.SEED)
    utils.add_folds(with_valid=False)
    for i in range(config.FOLDS):
        print('FOLD: ', i)
        run(i)