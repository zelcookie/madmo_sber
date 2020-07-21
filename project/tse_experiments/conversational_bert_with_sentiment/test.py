import torch
import transformers
import config
from dataset import TweetDataset
from model import TweetModel
from engine import calculate_jaccard_score, eval_fn
import numpy as np
import pandas as pd
import utils
from tqdm import tqdm

def predict(df_test):
    device = torch.device("cuda")
    model_config = transformers.BertConfig.from_pretrained(config.BERT_PATH)
    model_config.output_hidden_states = True
    model = TweetModel(conf=model_config)
    model.load_state_dict(torch.load("model.bin"))
    model.to(device)

    test_dataset = TweetDataset(
            tweet=df_test.text.values,
            sentiment=df_test.sentiment.values,
            selected_text=df_test.selected_text.values
    )

    data_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=1
    )
    
    return eval_fn(data_loader, model, device)

if __name__=='__main__':
    df_test = pd.read_csv('../input/valid.csv')
    jaccard = predict(df_test)
    print(jaccard)
