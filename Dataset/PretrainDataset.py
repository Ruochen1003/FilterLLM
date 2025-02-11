from torch.utils.data import Dataset
import fsspec
import pandas as pd
import torch
import random

class PretrainDataset(Dataset):
    def __init__(self, tokenizer, content_file_path, interaction_file_path, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        #item content : item id , content

        # Load descriptions from text file
        assert content_file_path.endswith(".csv"), "we need to load from a csv file"
        self._data_process(content_file_path, interaction_file_path)

    def __len__(self):
        return len(self.data['item'])

    def __getitem__(self, idx):
        item_id = self.data['item'][idx]
        item_content = self.data['content'][idx]
        users = self.data['users'][idx]
        return item_id, item_content, users

    def _data_process(self, content_file_path, interaction_file_path):
        item_user_mapping = pd.read_csv(interaction_file_path).groupby('item')['user'].agg(list).to_dict() # item:[users]
        all_item_content = pd.read_csv(content_file_path)
        all_item_content = all_item_content.set_index('item_id')['item_content'].to_dict()
        item_ids = list(item_user_mapping.keys())
        users_id = list(item_user_mapping.values())
        user_word = self._convert_to_str(users_id)
        #user_word = self._convert_list_to_str(user_word)
        content = self._content_process(item_ids, all_item_content)
        self.data = {
            "content": content,
            "item": item_ids,
            "users": user_word,
        }

    def _content_process(self, item_ids, all_item_content):
        #Which users would be attracted to an item with the content?
        return ["Which users would be attracted to an item with the content \"" + all_item_content[item_id] + "\" ?" for item_id in item_ids]

    def collate_fn(self, batch):
        item_id, item_content, users = zip(*batch)
        shuffled_user = self._shaffer_user(users)
        users_word = self._convert_list_to_str(shuffled_user)

        # Encode and pad the prompt and main texts
        encoded_prompt = self.tokenizer.encode_batch(item_content)
        encoded_main = self.tokenizer.encode_batch(users_word)

        # Get the prompt IDs, main IDs, and attention masks
        item_content_ids = torch.tensor(encoded_prompt[0])
        users_ids = torch.tensor(encoded_main[0])
        attention_mask = torch.cat((torch.tensor(encoded_prompt[1]), torch.tensor(encoded_main[1])), dim=1)

        # Truncate main IDs and attention mask if total length exceeds the maximum length
        total_length = item_content_ids.size(1) + users_ids.size(1)

        if total_length > self.max_length:
            excess_length = total_length - self.max_length
            item_content_ids = item_content_ids[:, :-excess_length]
            attention_mask = torch.cat((torch.tensor(encoded_prompt[1][:, :-excess_length]), torch.tensor(encoded_main[1])), dim=1)

        return item_id, item_content_ids, users_ids, attention_mask

    def _convert_to_str(self, data):
        if isinstance(data, list):
            return [self._convert_to_str(item) for item in data]  # 递归处理列表中的每个元素
        else:
            return 'user_' + str(data)

    def _convert_list_to_str(self, list):
        str_list = []
        for inner_list in list:
            random.shuffle(inner_list)
            str_list.append(''.join(inner_list))
        return str_list

    def _shaffer_user(self, user_list):
        return list(map(lambda x: random.shuffle(x) or x, list(user_list)))





# content_file_path = "D:\\PyProgram\\ListLLMRec\\data\\CiteULike\\item_content.csv"
# interaction_file_path = "D:\\PyProgram\\ListLLMRec\\data\\CiteULike\\warm_emb.csv"
#
# pretrain_dataset = PretrainDataset(0, content_file_path, interaction_file_path)