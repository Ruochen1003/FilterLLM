import os
from typing import Mapping, Any

import torch
import torch.nn as nn
from peft import PeftModel, get_peft_model
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



class LLMEncoder(nn.Module):
    """
    Encoder the input token sequence to sequence representation.
    """
    def __init__(self, config, llm):
        super(LLMEncoder, self).__init__()
        self.num_users = config.num_users
        self.num_items = config.num_items
        self.vocab_size = config.vocab_size
        self.config = config

        # Create new token embeddings for user/item tokens
        self.user_embeddings = nn.Embedding(self.num_users, config.hidden_size)
        self.item_embeddings = nn.Embedding(self.num_items, config.hidden_size)
        self.mapper = nn.Linear(config.hidden_size, config.hidden_size)
        self.relu = nn.ReLU()

        #TODO linear层的可扩展性

        # Randomly initialize the new token embeddings
        self.user_embeddings.weight.data.normal_(mean=0.0, std=config.initializer_range)

        # The pretrained language model
        self.llm = llm

    def forward(self, input_ids=None, set_position_ids=True, **kwargs):
        # Obtain the embeddings of the input id sequence
        # The input_embeds will be summed up with the pos_embed
        # And then forward into the transformer to get the results

        return self.llm(input_ids=input_ids, **kwargs)

    def map(self, input_emb):
        output_emb = self.mapper(input_emb)
        output_emb = self.relu(output_emb)
        return output_emb


class PairWiseRec(nn.Module):
    def __init__(self, config, encoder):
        """

        """
        super(PairWiseRec, self).__init__()

        # Obtain the number of users, items, and vocabulary size
        self.num_users = config.num_users
        self.num_items = config.num_items
        self.vocab_size = config.vocab_size

        # Base GPT model with extended user/item ID token embeddings
        self.encoder = encoder #将句子编码为embedding
        self.model_type = None

        # Tie the weights between the user embeddings and the user recommendation head

    def _compute_user_head_weight(self):
        origin_user_emb = self.encoder.user_embeddings.weight
        return origin_user_emb

    def user_head(self, input):
        linear_matrix = self._compute_user_head_weight()
        logist = torch.matmul(input, linear_matrix.T)
        return logist


    def forward(self, item_id, input_ids_prompt, attention_mask, labels, **kwargs):
        # Prompt embedding
        outputs_main = self.encoder(input_ids=input_ids_prompt, attention_mask=attention_mask, return_dict=True,
                                    output_hidden_states=True, **kwargs)
        last_hidden_states = outputs_main.hidden_states[-1]
        last_token_hidden_states = last_hidden_states[ :, -1, :]
        last_token_hidden_states = self.encoder.map(last_token_hidden_states)
        item_logist = self.user_head(last_token_hidden_states)
        multi_user_label = self.get_multi_hot_label(labels)
        if item_id == None:
            align_loss = 0
        else:
            item_emb = self.encoder.item_embeddings(item_id)
            align_loss = torch.mean((last_token_hidden_states - item_emb)**2)
        interaction_loss = F.cross_entropy(item_logist, multi_user_label.float())
        loss = 1 * align_loss + interaction_loss

        return loss, interaction_loss


    def load_user_emb(self, load_path):
        if load_path.split('.')[-1] == 'pt':
            weight = torch.load(load_path, map_location='cpu')
            self.encoder.user_embeddings.weight = torch.nn.Parameter(weight)

        if load_path.split('.')[-1] == 'npy':
            embeddings = np.load(load_path)
            user_embeddings = torch.tensor(embeddings[self.num_items:])
            self.encoder.user_embeddings.weight = torch.nn.Parameter(user_embeddings)

    def load_user_emb2(self, load_path):
        if load_path.split('.')[-1] == 'pt':
            weight = torch.load(load_path, map_location='cpu')
            self.encoder.user_embeddings.weight = torch.nn.Parameter(weight)

        if load_path.split('.')[-1] == 'npy':
            embeddings = np.load(load_path)
            user_embeddings = torch.tensor(embeddings[:self.num_users])
            self.encoder.user_embeddings.weight = torch.nn.Parameter(user_embeddings)

    def load_item_emb(self, load_path):
        if load_path.split('.')[-1] == 'pt':
            weight = torch.load(load_path, map_location='cpu')
            self.encoder.item_embeddings.weight = torch.nn.Parameter(weight)

        if load_path.split('.')[-1] == 'npy':
            embeddings = np.load(load_path)
            item_embeddings = torch.tensor(embeddings[:self.num_items])
            self.encoder.item_embeddings.weight = torch.nn.Parameter(item_embeddings)

    def load_item_emb2(self, load_path):
        if load_path.split('.')[-1] == 'pt':
            weight = torch.load(load_path, map_location='cpu')
            self.encoder.item_embeddings.weight = torch.nn.Parameter(weight)

        if load_path.split('.')[-1] == 'npy':
            embeddings = np.load(load_path)
            item_embeddings = torch.tensor(embeddings[self.num_users:])
            self.encoder.item_embeddings.weight = torch.nn.Parameter(item_embeddings)

    def from_pretrain(self, load_path):
        custom_weights = torch.load(os.path.join(load_path, "custom_weights.pt"))
        self.encoder.mapper.load_state_dict(custom_weights["mapper"])
        self.encoder.user_embeddings.load_state_dict(custom_weights["user_embeddings"])
        self.encoder.llm = PeftModel.from_pretrained(self.encoder.llm, load_path)
        # last_layer_path = load_path + '/last_layer.pt'
        # last_layer = self.encoder.llm.model.model.layers[-1]
        # last_layer.load_state_dict(torch.load(last_layer_path))



    def topk_predict(self,
                     input_ids=None,
                     attention_mask=None,
                     **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True,
                                    output_hidden_states=True, **kwargs)
        last_hidden_states = outputs.hidden_states[-1]
        last_token_hidden_states = last_hidden_states[ :, -1, :]
        last_token_hidden_states = self.encoder.map(last_token_hidden_states)
        logist = self.user_head(last_token_hidden_states)
        #logist = torch.softmax(logist, dim=-1)
        topk_user = torch.topk(logist, 20)
        return topk_user


    def get_labels(self, input_ids_main):
        labels = input_ids_main - self.vocab_size + 1
        labels[labels < 0] = 0
        multi_label = torch.zeros((len(input_ids_main), self.num_users + 1), dtype=torch.bool).to(input_ids_main.device)
        multi_label.scatter_(1, labels, True)
        multi_label[:,0] = False

        termination_label = torch.zeros((len(input_ids_main), self.num_users + 1), dtype=torch.bool).to(input_ids_main.device)
        termination_label[torch.arange(len(input_ids_main)), 0] = True

        # Tensor broadcaste
        multi_label = multi_label.unsqueeze(1)
        multi_label = multi_label.expand(-1, input_ids_main.shape[1], -1)

        termination_label = termination_label.unsqueeze(1)
        termination_label = termination_label.expand(-1, input_ids_main.shape[1], -1)
        return multi_label, termination_label

    def get_attention_mask(self, input_ids_main):
        # input_ids_main mask
        main_token_mask = input_ids_main > 0
        # input_ids_main 的最后一个
        last_main_token_index = main_token_mask.sum(dim=-1)

        last_main_token_mask = torch.zeros_like(input_ids_main, dtype=torch.bool, device=input_ids_main.device)
        user_token_mask = main_token_mask.clone().bool()

        last_main_token_mask[torch.arange(len(input_ids_main)), last_main_token_index - 1] = True
        user_token_mask[last_main_token_mask] = False

        return user_token_mask, last_main_token_mask

    def set_model_type(self, model_type):
        self.model_type = model_type
        self.encoder.model_type = model_type

    def lora_finetune(self, lora_config):
        self.encoder.llm = get_peft_model(self.encoder.llm, lora_config)

    def get_multi_hot_label(self, labels):
        labels = labels + 1
        multi_label = torch.zeros((len(labels), self.num_users + 1), dtype=torch.bool).to(labels.device)
        multi_label.scatter_(1, labels, True)
        multi_label = multi_label[:, 1:]
        return multi_label



























