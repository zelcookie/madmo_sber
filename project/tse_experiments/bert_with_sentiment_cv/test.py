import torch
import transformers
import config
from dataset import TweetDataset
from model import TweetModel
from engine import calculate_jaccard_score
import numpy as np
import pandas as pd
import utils
from tqdm import tqdm

def predict(df_test):
    device = torch.device("cuda")
    model_config = transformers.BertConfig.from_pretrained(config.BERT_PATH)
    model_config.output_hidden_states = True
    model1 = TweetModel(conf=model_config)
    model1.to(device)
    model1.load_state_dict(torch.load("model_0.bin"))
    model1.eval()

    model2 = TweetModel(conf=model_config)
    model2.to(device)
    model2.load_state_dict(torch.load("model_1.bin"))
    model2.eval()

    model3 = TweetModel(conf=model_config)
    model3.to(device)
    model3.load_state_dict(torch.load("model_2.bin"))
    model3.eval()

    model4 = TweetModel(conf=model_config)
    model4.to(device)
    model4.load_state_dict(torch.load("model_3.bin"))
    model4.eval()

    model5 = TweetModel(conf=model_config)
    model5.to(device)
    model5.load_state_dict(torch.load("model_4.bin"))
    model5.eval()

    final_output = []

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
    jaccards = utils.AverageMeter()
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for bi, d in enumerate(tk0):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            sentiment = d["sentiment"]
            orig_selected = d["orig_selected"]
            orig_tweet = d["orig_tweet"]
            targets_start = d["targets_start"]
            targets_end = d["targets_end"]
            offsets = d["offsets"].numpy()

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets_start = targets_start.to(device, dtype=torch.long)
            targets_end = targets_end.to(device, dtype=torch.long)

            outputs_start1, outputs_end1 = model1(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            
            outputs_start2, outputs_end2 = model2(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            
            outputs_start3, outputs_end3 = model3(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            
            outputs_start4, outputs_end4 = model4(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            
            outputs_start5, outputs_end5 = model5(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            outputs_start = (
                outputs_start1 
                + outputs_start2 
                + outputs_start3 
                + outputs_start4 
                + outputs_start5
            ) / 5
            outputs_end = (
                outputs_end1 
                + outputs_end2 
                + outputs_end3 
                + outputs_end4 
                + outputs_end5
            ) / 5
            
            outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
            outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
            
            jaccard_scores = []
            for px, tweet in enumerate(orig_tweet):
                selected_tweet = orig_selected[px]
                tweet_sentiment = sentiment[px]
                jaccard_score, output_sentence = calculate_jaccard_score(
                    original_tweet=tweet,
                    target_string=selected_tweet,
                    sentiment_val=tweet_sentiment,
                    idx_start=np.argmax(outputs_start[px, :]),
                    idx_end=np.argmax(outputs_end[px, :]),
                    offsets=offsets[px]
                )
                jaccard_scores.append(jaccard_score)
                final_output.append(output_sentence)
            jaccards.update(np.mean(jaccard_scores), ids.size(0))
    return final_output, jaccards.avg


if __name__=='__main__':
    df_test = pd.read_csv(config.VALID_FILE)
    predicts, jaccard = predict(df_test)
    print(jaccard)
