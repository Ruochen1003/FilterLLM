from torch.utils.data import Dataset
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import DataCollatorWithPadding

class RecDataset(Dataset):
    def __init__(self, content_file_path, interaction_file_path_list, dataset_name, max_length=512):
        super().__init__()
        self.max_length = max_length
        self.dataset_name = dataset_name

        # Load and process data
        assert content_file_path.endswith(".csv"), "Content file must be a CSV file"
        self._data_process(content_file_path, interaction_file_path_list)

    def __len__(self):
        return len(self.data['item'])

    def __getitem__(self, idx):
        item_id = self.data['item'][idx]
        users = self.data['users'][idx]
        prompt = self.data['prompt'][idx]

        return {
            "item_id": item_id,
            "labels": users,
            "input_ids_prompt": prompt,
            "attention_mask":None
        }

    def _data_process(self, content_file_path, interaction_file_path_list):
        item_ids = []
        user_id_list = []

        # Load interaction data
        for interaction_file_path in interaction_file_path_list:
            item_user_mapping = pd.read_csv(interaction_file_path).groupby('item')['user'].agg(list).to_dict()
            item_ids.extend(item_user_mapping.keys())
            user_id_list.extend(item_user_mapping.values())

        all_item_content = pd.read_csv(content_file_path)
        users_id_list = [torch.tensor(sublist) for sublist in user_id_list]
        prompt_list = self._item_content_process(item_ids, all_item_content)

        self.data = {
            "item": item_ids,
            "users": users_id_list,
            "prompt": prompt_list,
        }

    def _item_content_process(self, item_ids, all_item_content):
        prompt_list = []
        if self.dataset_name == 'CiteULike':
            for item_id in item_ids:
                prompt_list.append(
                    f"Assume you are an expert in the field of recommendation. "
                    f"Please predict which readers a paper titled \"{all_item_content.loc[item_id].title}\" would attract."
                )
        elif self.dataset_name == 'Amazon_Beauty':
            for item_id in item_ids:
                title = f"title: {all_item_content.loc[item_id].title}" if pd.notna(all_item_content.loc[item_id].title) else ""
                brand = f"brand: {all_item_content.loc[item_id].brand}" if pd.notna(all_item_content.loc[item_id].brand) else ""
                description = f"description: {all_item_content.loc[item_id].description}" if pd.notna(all_item_content.loc[item_id].description) else ""
                prompt_list.append(
                    f"A beauty product has the following features: {title}, {brand}, {description}. "
                    f"Predict the potential customer for this product."
                )
        elif self.dataset_name == 'ml-10m':
            for item_id in item_ids:
                title = f"Title: {all_item_content.loc[item_id].Title}" if pd.notna(all_item_content.loc[item_id].Title) else ""
                genres = f"Genres: {all_item_content.loc[item_id].Genres}" if pd.notna(all_item_content.loc[item_id].Genres) else ""
                prompt_list.append(
                    f"Assume you are an expert in the field of recommendation. "
                    f"A movie has the following features: {title}, {genres}. Predict the potential customer for this movie."
                )
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        return prompt_list


class RecDataCollator(DataCollatorWithPadding):
    def __init__(self, tokenizer, max_length=512, padding_value=-1):
        super().__init__(tokenizer)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding_value = padding_value

    def __call__(self, features):

        prompts = [f["input_ids_prompt"] for f in features]
        labels = [f["labels"] for f in features]

        # Tokenize prompts
        tokenized = self.tokenizer(
            prompts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # Pad labels
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=self.padding_value)

        # Build the batch
        batch = {
            "input_ids_prompt": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": padded_labels,
        }


        # Include item_id if needed
        if "item_id" in features[0]:
            batch["item_id"] = torch.tensor([f["item_id"] for f in features])


        return batch