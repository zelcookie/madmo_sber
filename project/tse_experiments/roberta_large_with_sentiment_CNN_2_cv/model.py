import config
import transformers
import torch.nn as nn
import torch

class TweetModel(transformers.BertPreTrainedModel):
    def __init__(self, conf):
        super(TweetModel, self).__init__(conf)
        self.roberta = transformers.RobertaModel.from_pretrained(config.ROBERTA_PATH, config=conf)
        self.drop_out = nn.Dropout(0.1)
        self.CNN_1 = torch.nn.Conv1d(1024 * 2, 128, 2)
        self.CNN_2 = torch.nn.Conv1d(128, 64, 2)
        self.l0 = nn.Linear(64, 2)
        nn.init.normal_(self.l0.weight, std=0.02)
    
    def forward(self, ids, mask, token_type_ids):
        _, _, out = self.roberta(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        out = torch.cat((out[-1], out[-2]), dim=-1)
        out = self.drop_out(out)
        out = torch.nn.functional.pad(out.transpose(1,2), (1, 0))
        out = self.CNN_1(out)
        out = torch.nn.LeakyReLU()(out)
        out = torch.nn.functional.pad(out, (1, 0))
        out = self.CNN_2(out).transpose(1,2)
        logits = self.l0(out)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits


