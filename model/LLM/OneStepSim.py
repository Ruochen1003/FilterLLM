import os
from typing import Mapping, Any, Union, Optional, Callable

import torch
import torch.nn as nn
from peft import PeftModel, get_peft_model
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import PreTrainedModel, LlamaPreTrainedModel



class OneStepRec(PreTrainedModel):
    def __init__(self, config, encoder, num_users, num_items):
        super().__init__(config)
        # Obtain the number of users, items, and vocabulary size
        self.num_users = num_users
        self.num_items = num_items
        self.vocab_size = config.vocab_size
        self.user_embeddings = nn.Embedding(self.num_users, config.hidden_size)
        self.item_embeddings = nn.Embedding(self.num_items, config.hidden_size)
        self.mapper = nn.Linear(config.hidden_size, config.hidden_size)
        self.relu = nn.ReLU()

        # Base GPT model with extended user/item ID token embeddings
        self.encoder = encoder #将句子编码为embedding
        self.lamda = config.lamda
        #self.lora_finetune(lora_config)

        for name, param in self.named_parameters():
            if ("user_embeddings" not in name) and ("mapper" not in name) and ('lora' not in name):
                # if ("mapper" not in name):
                param.requires_grad = False

        # Tie the weights between the user embeddings and the user recommendation head

    def _compute_user_head_weight(self):
        origin_user_emb = self.user_embeddings.weight
        return origin_user_emb

    def map(self, input_emb):
        output_emb = self.mapper(input_emb)
        output_emb = self.relu(output_emb)
        return output_emb

    def user_head(self, input):
        linear_matrix = self._compute_user_head_weight()
        logist = torch.matmul(input, linear_matrix.T)
        return logist


    def forward(self, item_id, input_ids_prompt, attention_mask, labels = None, **kwargs):
        # Prompt embedding
        outputs_main = self.encoder(input_ids=input_ids_prompt, attention_mask=attention_mask, return_dict=True,
                                    output_hidden_states=True, **kwargs)
        last_hidden_states = outputs_main.hidden_states[-1]
        last_token_hidden_states = last_hidden_states[ :, -1, :]
        last_token_hidden_states = self.map(last_token_hidden_states)
        item_logist = self.user_head(last_token_hidden_states)

        multi_user_label = self.get_multi_hot_label(labels)
        n = multi_user_label.sum(dim=-1, keepdim=True)
        item_emb = self.item_embeddings(item_id)
        align_loss = torch.mean((last_token_hidden_states - item_emb)**2)

        interaction_loss = F.cross_entropy(item_logist, multi_user_label.float()/n)
        loss = interaction_loss + self.lamda * align_loss
        _,topk_user = torch.topk(item_logist, 20)

        return {'loss':loss, 'logits':item_logist, 'topk_user':topk_user, 'item_id': item_id}

    # def forward(self, item_id, input_ids_prompt, attention_mask, labels, **kwargs):
    #     # Prompt embedding
    #     outputs_main = self.encoder(input_ids=input_ids_prompt, attention_mask=attention_mask, return_dict=True,
    #                                 output_hidden_states=True, **kwargs)
    #     last_hidden_states = outputs_main.hidden_states[-1]
    #     last_token_hidden_states = last_hidden_states[ :, -1, :]
    #     last_token_hidden_states = self.map(last_token_hidden_states)
    #     item_logist = self.user_head(last_token_hidden_states)
    #     multi_user_label = self.get_multi_hot_label(labels)
    #
    #     item_emb = self.item_embeddings(item_id)
    #     align_loss = torch.mean((last_token_hidden_states - item_emb)**2)
    #     interaction_loss = F.cross_entropy(item_logist, multi_user_label.float())
    #     loss = 10 * align_loss + interaction_loss
    #
    #     return {'loss':loss}

    def load_user_emb(self, load_path):
        if load_path.split('.')[-1] == 'pt':
            weight = torch.load(load_path, map_location='cpu')
            self.user_embeddings.weight = torch.nn.Parameter(weight)

        if load_path.split('.')[-1] == 'npy':
            embeddings = np.load(load_path)
            user_embeddings = torch.tensor(embeddings[self.num_items:])
            self.user_embeddings.weight = torch.nn.Parameter(user_embeddings)
            self.num_users = int(self.num_users)

    def load_user_emb2(self, load_path):
        if load_path.split('.')[-1] == 'pt':
            weight = torch.load(load_path, map_location='cpu')
            self.user_embeddings.weight = torch.nn.Parameter(weight)

        if load_path.split('.')[-1] == 'npy':
            embeddings = np.load(load_path)
            user_embeddings = torch.tensor(embeddings[:self.num_users])
            self.user_embeddings.weight = torch.nn.Parameter(user_embeddings)

    def load_item_emb(self, load_path):
        if load_path.split('.')[-1] == 'pt':
            weight = torch.load(load_path, map_location='cpu')
            self.item_embeddings.weight = torch.nn.Parameter(weight)

        if load_path.split('.')[-1] == 'npy':
            embeddings = np.load(load_path)
            item_embeddings = torch.tensor(embeddings[:self.num_items])
            self.item_embeddings.weight = torch.nn.Parameter(item_embeddings)
            self.num_items = int(self.num_items)

    def load_item_emb2(self, load_path):
        if load_path.split('.')[-1] == 'pt':
            weight = torch.load(load_path, map_location='cpu')
            self.item_embeddings.weight = torch.nn.Parameter(weight)

        if load_path.split('.')[-1] == 'npy':
            embeddings = np.load(load_path)
            item_embeddings = torch.tensor(embeddings[self.num_users:])
            self.item_embeddings.weight = torch.nn.Parameter(item_embeddings)

    def lora_finetune(self, lora_config):
        self.encoder = get_peft_model(self.encoder, lora_config)

    def get_multi_hot_label(self, labels):
        labels = labels + 1
        multi_label = torch.zeros((len(labels), self.num_users + 1), dtype=torch.bool).to(labels.device)
        multi_label.scatter_(1, labels, True)
        multi_label = multi_label[:, 1:]
        return multi_label

    # def save_pretrained(self, save_directory, **kwargs):
    #     """
    #     自定义保存逻辑，仅保存 LoRA 层、mapper 和 user_embeddings。
    #     """
    #     os.makedirs(save_directory, exist_ok=True)
    #
    #     # 保存 LoRA 层
    #     if isinstance(self.encoder, PeftModel):
    #         self.encoder.save_pretrained(save_directory, **kwargs)
    #
    #     # 保存 mapper 和 user_embeddings
    #     torch.save(
    #         {
    #             "mapper": self.mapper.state_dict(),
    #             "user_embeddings": self.user_embeddings.state_dict(),
    #         },
    #         os.path.join(save_directory, "custom_weights.pt"),
    #     )
    #
    # @classmethod
    # def from_pretrained(cls, save_directory, config, lora_config, encoder, num_users, num_items):
    #     """
    #     自定义加载逻辑。
    #     """
    #     model = cls(config, lora_config, encoder, num_users, num_items)
    #
    #     # 加载 LoRA 层
    #     if isinstance(model.encoder, PeftModel):
    #         model.encoder = PeftModel.from_pretrained(model.encoder, save_directory)
    #
    #     # 加载 mapper 和 user_embeddings
    #     custom_weights = torch.load(os.path.join(save_directory, "custom_weights.pt"))
    #     model.mapper.load_state_dict(custom_weights["mapper"])
    #     model.user_embeddings.load_state_dict(custom_weights["user_embeddings"])
    #
    #     return model

    def topk_predict(self,
                     input_ids_prompt=None,
                     attention_mask=None,
                     ):
        outputs = self.encoder(input_ids=input_ids_prompt, attention_mask=attention_mask, return_dict=True,
                                    output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        last_token_hidden_states = last_hidden_states[ :, -1, :]
        last_token_hidden_states = self.encoder.map(last_token_hidden_states)
        logist = self.user_head(last_token_hidden_states)
        topk_user = torch.topk(logist, 20)
        return topk_user



























