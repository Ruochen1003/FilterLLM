import pickle
import argparse

import numpy as np
import torch
import torch.optim as optim
import os
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import LlamaForCausalLM,get_scheduler,AutoTokenizer, TrainingArguments, Trainer,TrainerCallback

from Dataset.RecDataset import RecDataset, RecDataCollator

from model.LLM.OneStepSim import OneStepRec
from tqdm import tqdm

import pandas as pd
from peft import LoraConfig, TaskType, get_peft_model


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
    rec_model = OneStepRec(config, LlamaModel, num_users, num_items)
    rec_model.encoder = get_peft_model(LlamaModel, lora_config)

    user_emb_path = os.path.join('./model_weight', args.dataset, 'base_model', f'{args.backbone_type}_{args.LLM_type}.npy')
    accelerator.print(f"Load user embedding from {user_emb_path}")
    rec_model.load_user_emb(user_emb_path)
    item_emb_path = os.path.join('./model_weight', args.dataset, 'base_model',
                                 f'{args.backbone_type}_{args.LLM_type}.npy')
    accelerator.print(f"Load item embedding from {item_emb_path}")
    rec_model.load_item_emb(item_emb_path)
    accelerator.print("Success!")
    accelerator.print("-----End Instantiating the Pairwise Recommendation Model-----\n")

    # '''
    #     Freeze the parameters of the pretrained GPT2 for content model
    # '''
    # # TODO need extension for llama, here only allows user embedding to be trained

    '''
        Define the review pretrain data generator
    '''
    # TODO generate the pretrain data with correct format (pretrain:raw item content, interacted users)

    rec_model.train()

    accelerator.print("-----Pretrain Trainable Parameters-----")
    for name, param in rec_model.named_parameters():
        if param.requires_grad:
            accelerator.print("{} : {}".format(name, param.shape))

    accelerator.print("\n-----Pretrain Non-trainable Parameters-----")
    for name, param in rec_model.named_parameters():
        if not param.requires_grad:
            accelerator.print("{} : {}".format(name, param.shape))


    content_file_path = os.path.join(dataset_root, 'raw-data.csv')
    interaction_file_path = os.path.join(dataset_root, 'warm_emb.csv')
    val_file_path = os.path.join(dataset_root, 'cold_item_val.csv')
    accelerator.print("-----Preparing Pretrain Dataset-----")

    train_dataset = RecDataset(content_file_path, [interaction_file_path], args.dataset, max_length=args.max_length)
    val_dataset = RecDataset(content_file_path, [val_file_path], args.dataset,
                                       max_length=args.max_length)

    training_args = TrainingArguments(
        seed=args.seed,
        per_device_train_batch_size=args.pretrain_batch_size,
        per_device_eval_batch_size=args.predict_batch_size,
        warmup_ratio=0.05,
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        output_dir=f"./results/{args.dataset}/{args.LLM_type}_{args.lamda}",
        save_total_limit=1,
        load_best_model_at_end=False,
        report_to=None,
        eval_delay=1,
    )

    trainer = Trainer(
        model=rec_model,
        args = training_args,
        data_collator = data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    # output = trainer.predict(val_dataset)
    # breakpoint()

    trainer.train(resume_from_checkpoint = None)
    llm_weight_path = os.path.join('./model_weight', args.dataset, 'LLM')
    load_path = os.path.join(llm_weight_path, 'lora_finetune', f"{args.LLM_type}_{args.backbone_type}_{args.lamda}")
    #trainer.save_model(load_path)
    rec_model.encoder.save_pretrained(load_path)
    torch.save(
        {
            "mapper": rec_model.mapper.state_dict(),
            "user_embeddings": rec_model.user_embeddings.state_dict(),
        },
        os.path.join(load_path, "custom_weights.pt"),
    )



    #
    # """
    #     Predict the interaction and update the dataset
    # """
    #
    # accelerator.print("-----Predict Interaction for Cold Item-----")
    # rec_model.eval()
    # dataset_root = os.path.join('./data', args.dataset)
    # content_file_path = os.path.join(dataset_root, 'raw-data.csv')
    # cold_val_file_path = os.path.join(dataset_root, 'cold_item_test.csv')
    # cold_test_file_path = os.path.join(dataset_root, 'cold_item_val.csv')
    # #interaction_file_path = os.path.join(dataset_root, 'warm_emb.csv')
    # # cold_item_dataset = PretrainDataset(tokenizer, content_file_path, interaction_file_path,args.dataset)
    # cold_item_dataset = RecDataset(tokenizer, content_file_path, [cold_val_file_path, cold_test_file_path],
    #                                     args.dataset)
    #
    # llm_weight_path = os.path.join('./model_weight', args.dataset, 'LLM')
    # if args.full_model_update == 'lora finetune':
    #     load_path = os.path.join(llm_weight_path, 'lora_finetune', f"{args.LLM_type}_{args.backbone_type}")
    #     rec_model.from_pretrain(load_path)
    #     accelerator.print(f"load lora finetune model from {load_path}")
    # elif args.full_model_update == 'full finetune':
    #     load_path = os.path.join(llm_weight_path, 'full_finetune', f"{args.LLM_type}_{args.backbone_type}")
    #     rec_model.encoder.llm.from_pretrain(load_path)
    # else:
    #     raise Exception('full_model_update must be chosen from lora finetune or full finetune')
    #
    # accelerator.print(f"use model in {load_path}")
    #
    # #print(f"Model is on device: {next(pairwise_rec.parameters()).device}")
    # item_list, predict_user_list = topk_predict(cold_item_predict_dataloader, pairwise_rec, accelerator)
    # #test(cold_item_predict_dataloader, pairwise_rec,accelerator)
    #
    # accelerator.print("-----Prediction Finish-----")
    #
    # accelerator.print("-----Updating dataset-----")
    # update_dataset(item_list, predict_user_list, accelerator)
    #
    # accelerator.print("-----Success-----")






def update_dataset(item_list, predict_user_list, accelerator):
    # 创建一个空的列表来存储 user-item pairs

    # 仅主进程负责文件写入
    if accelerator.is_main_process:
        user_item_pairs = []

        for batch_item_list, batch_predict_top_k in zip(item_list, predict_user_list):
            batch_predict_user_list = batch_predict_top_k[1]
            batch_predict_value_list = batch_predict_top_k[0]
            for item_id, predict_users in zip(batch_item_list, batch_predict_user_list):
                for user_id in predict_users:
                    user_item_pairs.append([user_id.item(), item_id.item()])

        # 将数据转换为 DataFrame
        df = pd.DataFrame(user_item_pairs, columns=["user", "item"])

        # 保存路径
        dataset_root = os.path.join('./data', args.dataset)
        saved_file_path = os.path.join(dataset_root, f'{args.LLM_type}predicted_cold_item_interaction.csv')

        df.to_csv(saved_file_path, index=False)
        print(f"User-item pairs saved to {saved_file_path}")


#TODO set max_length input of LLM to be adjusted

# other setting
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random Seed.')
#parser.add_argument('--gpu_id', type=str, default='4,5')

# dataset setting
parser.add_argument("--dataset", type=str, default='ml-10m', help="specify the dataset for experiment")

# LLM setting
parser.add_argument('--LLM_root', type=str, default="./LLM_base")
parser.add_argument('--LLM_type', type=str, default="Llama2-7B", help='LLM model type (Llama2-7B, GPT2, Llama3-1B, Llama3-3B,  Llama3-13B)')
parser.add_argument("--max_length", type= int, default=512, help='max taken length of LLM')

# recommendation setting
parser.add_argument('--list_length', type=int, default=40, help='number of retrivaled users')

# training details
parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
parser.add_argument('--pretrain_batch_size', type=int, default=16)
parser.add_argument('--finetune_batch_size', type=int, default=1)
parser.add_argument('--predict_batch_size',  type=int, default=2)

# backbone
parser.add_argument('--backbone_type', type=str, default='LightGCN')
parser.add_argument('--hidden_layers', type=list, default=[512, 256, 512], help='hidden_layers of mapper')

parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--lamda', type=int, default=10)

parser.add_argument('--pretrain', type=int, default=1, help='0 for predict only, 1 for pretraining only, 2 for fine-tuning only, 3 for both')

parser.add_argument('--predict_way', type=str, default='positive', help='positive, negative, both')
parser.add_argument('--full_model_update', type=str, default='lora finetune', help='lora finetune or full finetune')


#TODO


args = parser.parse_args()

if __name__ == "__main__":
    main(args)
