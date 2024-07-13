import torch
import torch.nn as nn
from transformers import BertModel
from IntuitionNN import IntuitionNN

class BERTIntuitionModel(nn.Module):
    def __init__(self, num_labels):
        super(BERTIntuitionModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.intuition_nn = IntuitionNN(768, [512, 256, 128], 128, num_labels)  # Adjust intuition_size to match the last layer

    def forward(self, input_ids, attention_mask, iteration):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.intuition_nn(pooled_output, iteration)
