import config
import transformers
import torch.nn as nn
import torch

class TweetModel(transformers.BertPreTrainedModel):
    def __init__(self, conf):
        super(TweetModel, self).__init__(conf)
        self.roberta = transformers.RobertaModel.from_pretrained(config.ROBERTA_PATH, config=conf)
        self.drop_out = nn.Dropout(0.5)
        self.l0 = nn.Linear(768, 2)
        nn.init.normal_(self.l0.weight, std=0.02)
    
    def forward(self, ids, mask):#, token_type_ids):
        _, _, out = self.roberta(
            ids,
            attention_mask=mask,
            #token_type_ids=token_type_ids
        )

        # # out_1 = torch.cat((out[-1], out[-2]), dim=1)
        # # print(out_1.shape)
        # # out_1 = torch.mean(out_1, 0)
        # # print(out_1.shape)
        # # out_2 = torch.cat((out[-3], out[-4]), dim=-1)
        # # out_2 = torch.mean(out_2, 0)
        # out = torch.cat((out[-1]+out[-2], out[-3]+out[-4]), dim=-1)
        # out = self.drop_out(out)
        # logits = self.l0(out)
        x = torch.stack([out[-1], out[-2], out[-3], out[-4]])
        x = torch.mean(x, 0)
        x = self.drop_out(x)
        logits = self.l0(x)
        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits


