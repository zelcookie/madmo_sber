import config
from dataset import TweetDataset
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


EPOCHS = 5
MAX_EPOCHS = 3


def run():
    seed_everything(config.SEED)
    df_train = pd.read_csv(config.TRAINING_FILE).dropna().reset_index(drop=True)
    
    
    train_dataset = TweetDataset(
        tweet=df_train.text.values,
        sentiment=df_train.sentiment.values,
        selected_text=df_train.selected_text.values
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=4
    )

    device = torch.device("cuda")
    model_config = transformers.BertConfig.from_pretrained(config.BERT_PATH)
    model_config.output_hidden_states = True
    model = TweetModel(conf=model_config)
    model.to(device)

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * EPOCHS)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=num_train_steps
    )

    es = utils.EarlyStopping(patience=2, mode="max")
    
    for epoch in range(EPOCHS):
        engine.train_fn(train_data_loader, model, optimizer, device, scheduler=scheduler)
        if epoch+1==MAX_EPOCHS:
            torch.save(model.state_dict(), 'model_full.bin')
            break
            


if __name__=='__main__':
    run()