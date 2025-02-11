import pickle
import argparse

import numpy as np
import torch
import torch.optim as optim
import os
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import LlamaForCausalLM,get_scheduler,AutoTokenizer, TrainingArguments, Trainer

from Dataset.RecDataset import RecDataset, RecDataCollator

from model.LLM.OneStepSim import OneStepRec
from tqdm import tqdm

import pandas as pd
from peft import LoraConfig, TaskType, PeftModel


def main(args):
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    torch.cuda.manual_seed_all(args.seed)

    accelerator = Accelerator(cpu=False)
    if torch.cuda.is_available():
        print(f"device: {accelerator.device}")
        print(f"Available GPUs: {torch.cuda.device_count()}")
    else:
        print("CUDA is not available.")

    accelerator.print("-----Current Setting-----")
    accelerator.print(f"dataset: {args.dataset}")

    num_gpus = torch.cuda.device_count()
    accelerator.print(f"num_gpus: {num_gpus}")
    accelerator.print(f'process: {accelerator.num_processes}')

    #TODO complete the output training information,user number, item number
    accelerator.print("-----Begin Obtaining Dataset Info-----")
    dataset_root = os.path.join('./data', args.dataset)
    num_user_item_path = os.path.join(dataset_root, 'convert_dict.pkl')
    convert_dict = pickle.load(open(num_user_item_path,'rb'))
    num_users= max(convert_dict['user_array']) + 1
    num_items= max(convert_dict['item_array']) + 1
    origin_dim = 200

    '''
        Obtain the tokenizer with user/item tokens
    '''
    accelerator.print("-----Begin Obtaining the Tokenizer-----")
    if args.LLM_type =='LLama2':
        llm_root = os.path.join(args.LLM_root, args.LLM_type)
    elif args.LLM_type == 'Llama3-1B':
        llm_root = '/home/models/Llama-3.2-1B'
    elif args.LLM_type == "Llama3-3B":
        llm_root = '/home/models/Llama-3.2-3B'
    elif args.LLM_type == "Llama2-7B":
        llm_root = '/home/models/Llama-2-7b-hf'
    elif args.LLM_type == "Llama3-13B":
        llm_root = '/home/models/Llama-2-13b-hf'
    else:
        raise Exception("LLM_type don't exist")
    tokenizer = AutoTokenizer.from_pretrained(llm_root,padding_side = "left")
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = RecDataCollator(tokenizer)
    accelerator.print("Success!")
    accelerator.print("-----End Obtaining the Tokenizer-----\n")

    '''
        Instantiate the pretrained Llama model
    '''
    # TODO need extension for llama
    accelerator.print(f"-----Begin Instantiating the Pretrained {args.LLM_type} Model-----")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        inference_mode=False,
        task_type=TaskType.CAUSAL_LM
    )
    LlamaModel = LlamaForCausalLM.from_pretrained(llm_root, attn_implementation="eager")
    config = LlamaModel.config
    config.hidden_layers = args.hidden_layers
    # config.num_users = num_users
    # config.num_items = num_items
    config.origin_dim = origin_dim
    config.lamda = args.lamda
    accelerator.print("Success!")
    accelerator.print(f"-----End Instantiating the Pretrained {args.LLM_type} Model-----\n")
    '''
        Instantiate the llama for recommendation content model
    '''
    accelerator.print("-----Begin Instantiating the Pairwise Recommendation Model-----")



    accelerator.print("Success!")
    accelerator.print("-----End Instantiating the Pairwise Recommendation Model-----\n")
    inference_args = TrainingArguments(
        output_dir=f'data/{args.dataset}',
        do_train=False,
        do_predict=True,
        per_device_eval_batch_size=args.predict_batch_size,
        dataloader_drop_last=False,
        dataloader_num_workers=1,
        fp16_full_eval=True,
    )
    weight_path = os.path.join('model_weight', args.dataset, 'LLM', "lora_finetune", f"{args.LLM_type}_{args.backbone_type}_{args.lamda}")
    #rec_model = OneStepRec.from_pretrained(weight_path, config, lora_config, LlamaModel, num_users, num_items)
    rec_model = OneStepRec(config, LlamaModel, num_users, num_items)

    rec_model.encoder = PeftModel.from_pretrained(LlamaModel, weight_path)
    custom_weights = torch.load(os.path.join(weight_path, "custom_weights.pt"))
    rec_model.mapper.load_state_dict(custom_weights["mapper"])
    rec_model.user_embeddings.load_state_dict(custom_weights["user_embeddings"])

    dataset_root = os.path.join('./data', args.dataset)
    content_file_path = os.path.join(dataset_root, 'raw-data.csv')
    cold_val_file_path = os.path.join(dataset_root, 'cold_item_test.csv')
    cold_test_file_path = os.path.join(dataset_root, 'cold_item_val.csv')
    cold_item_dataset = RecDataset(content_file_path, [cold_val_file_path, cold_test_file_path],
                                   args.dataset, max_length=args.max_length)

    trainer = Trainer(model=rec_model, args=inference_args, data_collator = data_collator)

    output = trainer.predict(cold_item_dataset)
    update_dataset(output)


# class CustomTrainer(Trainer):
#     def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
#         # 在 prediction_step 中调用模型的 `topk_predict` 方法
#         # 假设模型中定义了 topk_predict 方法
#         outputs = model.topk_predict(**inputs)
#
#         # 只返回 topk 用户
#         topk_user = outputs.indices.tolist()  # 获取 top-k 用户的索引
#         return topk_user


def update_dataset(output):
    item_list = output.predictions[2]
    predict_user_list = output.predictions[1]

    user_item_pairs = []

    for batch_item_list, batch_predict_top_k in zip(item_list, predict_user_list):
        for user_id in batch_predict_top_k:
            user_item_pairs.append([user_id.item(), batch_item_list.item()])

    # 将数据转换为 DataFrame
    df = pd.DataFrame(user_item_pairs, columns=["user", "item"])

    # 保存路径
    dataset_root = os.path.join('./data', args.dataset)
    saved_file_path = os.path.join(dataset_root, f'{args.LLM_type}_predicted_cold_item_interaction_{args.lamda}.csv')

    df.to_csv(saved_file_path, index=False)
    print(f"User-item pairs saved to {saved_file_path}")


# other setting
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random Seed.')
#parser.add_argument('--gpu_id', type=str, default='4,5')

# dataset setting
parser.add_argument("--dataset", type=str, default='CiteULike', help="specify the dataset for experiment")

# LLM setting
parser.add_argument('--LLM_root', type=str, default="./LLM_base")
parser.add_argument('--LLM_type', type=str, default="Llama2-7B", help='LLM model type (Llama2-7B, GPT2, Llama3-1B, Llama3-3B,  Llama3-13B)')
parser.add_argument("--max_length", type= int, default=512, help='max taken length of LLM')

# recommendation setting
parser.add_argument('--list_length', type=int, default=40, help='number of retrivaled users')

# training details
parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
parser.add_argument('--pretrain_batch_size', type=int, default=8)
parser.add_argument('--finetune_batch_size', type=int, default=1)
parser.add_argument('--predict_batch_size',  type=int, default=8)

# backbone
parser.add_argument('--backbone_type', type=str, default='LightGCN')
parser.add_argument('--hidden_layers', type=list, default=[512, 256, 512], help='hidden_layers of mapper')

parser.add_argument('--num_pretrained_epochs', type=int, default=100) #如果使用多分类，10轮就足够了
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--lamda', type=int, default=10)

parser.add_argument('--pretrain', type=int, default=1, help='0 for predict only, 1 for pretraining only, 2 for fine-tuning only, 3 for both')

parser.add_argument('--predict_way', type=str, default='positive', help='positive, negative, both')
parser.add_argument('--full_model_update', type=str, default='lora finetune', help='lora finetune or full finetune')


#TODO


args = parser.parse_args()

if __name__ == "__main__":
    main(args)
