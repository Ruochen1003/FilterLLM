import pickle
import argparse
import torch
import torch.optim as optim
import os
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import LlamaForCausalLM,get_scheduler,AutoTokenizer

from Dataset.PretrainDataset2 import PretrainDataset

from model.LLM.llama_listrec_CF2 import LLMEncoder, PairWiseRec
from tqdm import tqdm

import pandas as pd
from peft import LoraConfig, get_peft_model, TaskType


def main(args):
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    torch.cuda.manual_seed_all(args.seed)

    accelerator = Accelerator(cpu=False)
    device = accelerator.device
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
    LlamaModel = LlamaForCausalLM.from_pretrained(llm_root)
    config = LlamaModel.config
    config.hidden_layers = args.hidden_layers
    config.num_users = num_users
    config.num_items = num_items
    config.origin_dim = origin_dim
    accelerator.print("Success!")
    accelerator.print(f"-----End Instantiating the Pretrained {args.LLM_type} Model-----\n")

    '''
        Instantiate the llama for recommendation content model
    '''
    accelerator.print("-----Begin Instantiating the Pairwise Recommendation Model-----")
    content_encoder = LLMEncoder(config, LlamaModel)
    pairwise_rec = PairWiseRec(config, content_encoder)
    user_emb_path = os.path.join('./model_weight', args.dataset, 'base_model', f'{args.backbone_type}_{args.LLM_type}.npy')
    accelerator.print(f"Load user embedding from {user_emb_path}")
    pairwise_rec.load_user_emb(user_emb_path)
    #pairwise_rec.load_user_emb2(user_emb_path)
    item_emb_path = os.path.join('./model_weight', args.dataset, 'base_model',
                                 f'{args.backbone_type}_{args.LLM_type}.npy')
    accelerator.print(f"Load item embedding from {item_emb_path}")
    pairwise_rec.load_item_emb(item_emb_path)
    #pairwise_rec.load_item_emb2(item_emb_path)
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
    if args.pretrain == 1 or args.pretrain == 3:
        pairwise_rec.train()
        if args.full_model_update == 'lora finetune':
            for name, param in pairwise_rec.named_parameters():
                if ("user_embeddings" not in name) and ("mapper" not in name):
                #if ("mapper" not in name):
                    param.requires_grad = False
            pairwise_rec.lora_finetune(lora_config)
        elif args.full_model_update == 'full finetune':
            pass
        else:
            raise Exception('full_model_update must be chosen from lora finetune or full finetune')
        # llm_weight_path = os.path.join('./model_weight', args.dataset, 'LLM')
        # if args.full_model_update == 'lora finetune':
        #     load_path = os.path.join(llm_weight_path, 'lora_finetune', f"{args.LLM_type}_{args.backbone_type}_wo_item")
        #     accelerator.print(f'load weight from {load_path}')
        #     pairwise_rec.from_pretrain(load_path)
        #     # pairwise_rec.lora_finetune(lora_config)
        #     # pairwise_rec.load_state_dict(torch.load(load_path))
        #     # user_emb_path = load_path + '/user_embeddings.pt'
        #     # accelerator.unwrap_model(pairwise_rec).encoder.user_embeddings.load_state_dict(torch.load(user_emb_path))
        #     # mapper_path = load_path + '/mapper.pt'
        #     # accelerator.unwrap_model(pairwise_rec).encoder.mapper.load_state_dict(torch.load(mapper_path))
        #     # for name, param in pairwise_rec.named_parameters():
        #     #     if ("user_embeddings" in name) or ("mapper" in name) or ("lora" in name):
        #     #         accelerator.print(name, ":", param)
        #
        #     for name, param in pairwise_rec.named_parameters():
        #         param.requires_grad = True
        #     for name, param in pairwise_rec.named_parameters():
        #         if ("user_embeddings" not in name) and ("mapper" not in name) and ("lora" not in name):
        #             param.requires_grad = False
        # elif args.full_model_update == 'full finetune':
        #     load_path = os.path.join(llm_weight_path, 'full_finetune', f"{args.LLM_type}_{args.backbone_type}")
        #     accelerator.unwrap_model(pairwise_rec).encoder.llm.from_pretrained(load_path)
        #     mapper_path = load_path + '/mapper.pt'
        #     accelerator.unwrap_model(pairwise_rec).encoder.mapper.load_state_dict(torch.load(mapper_path))
        # else:
        #     raise Exception('full_model_update must be chosen from lora finetune or full finetune')
        optimizer = optim.AdamW(pairwise_rec.parameters(), lr=args.lr)
        scheduler = get_scheduler(
            name="cosine",  # 调度器类型
            optimizer=optimizer,
            num_warmup_steps=1285,  # 预热步数
            num_training_steps=25700,  # 总训练步数
        )

        accelerator.print("-----Pretrain Trainable Parameters-----")
        for name, param in pairwise_rec.named_parameters():
            if param.requires_grad:
                accelerator.print("{} : {}".format(name, param.shape))

        accelerator.print("\n-----Pretrain Non-trainable Parameters-----")
        for name, param in pairwise_rec.named_parameters():
            if not param.requires_grad:
                accelerator.print("{} : {}".format(name, param.shape))


        content_file_path = os.path.join(dataset_root, 'raw-data.csv')
        #interaction_file_path = os.path.join(dataset_root, 'warm_emb.csv')
        interaction_file_path = os.path.join(dataset_root, 'warm_val.csv')
        accelerator.print("-----Preparing Pretrain Dataset-----")

        pretrain_dataset = PretrainDataset(tokenizer, content_file_path, [interaction_file_path], args.dataset, max_length=args.max_length)
        pretrain_dataloader = DataLoader(pretrain_dataset,
                                         batch_size=args.pretrain_batch_size,
                                         collate_fn=pretrain_dataset.collate_fn)
        pairwise_rec, optimizer, pretrain_dataloader, scheduler = accelerator.prepare(
            pairwise_rec, optimizer, pretrain_dataloader, scheduler
        )

        accelerator.print("-----Begin Pretraining Loop-----")
        # TODO detail of the pretrain need to be check
        llm_weight_path = os.path.join('./model_weight', args.dataset, 'LLM')

        pretrain_loop(accelerator, pretrain_dataloader, pairwise_rec, optimizer,
                      device, args, llm_weight_path, scheduler)

        accelerator.print("-----End Pretraining Loop-----")

    """
        Predict the interaction and update the dataset
    """

    accelerator.print("-----Predict Interaction for Cold Item-----")
    pairwise_rec.eval()
    dataset_root = os.path.join('./data', args.dataset)
    content_file_path = os.path.join(dataset_root, 'raw-data.csv')
    cold_val_file_path = os.path.join(dataset_root, 'cold_item_test.csv')
    cold_test_file_path = os.path.join(dataset_root, 'cold_item_val.csv')
    #interaction_file_path = os.path.join(dataset_root, 'warm_emb.csv')
    # cold_item_dataset = PretrainDataset(tokenizer, content_file_path, interaction_file_path,args.dataset)
    cold_item_dataset = PretrainDataset(tokenizer, content_file_path, [cold_val_file_path, cold_test_file_path],
                                        args.dataset)
    # cold_item_dataset = PretrainDataset(tokenizer, content_file_path, [interaction_file_path],
    #                                     args.dataset)
    pretrain_dataloader = DataLoader(cold_item_dataset,
                                     batch_size=args.predict_batch_size,
                                     collate_fn=cold_item_dataset.collate_fn)
    llm_weight_path = os.path.join('./model_weight', args.dataset, 'LLM')
    if args.full_model_update == 'lora finetune':
        load_path = os.path.join(llm_weight_path, 'lora_finetune', f"{args.LLM_type}_{args.backbone_type}_{args.lamda}")
        pairwise_rec.from_pretrain(load_path)
        accelerator.print(f"load lora finetune model from {load_path}")
    elif args.full_model_update == 'full finetune':
        load_path = os.path.join(llm_weight_path, 'full_finetune', f"{args.LLM_type}_{args.backbone_type}")
        pairwise_rec.encoder.llm.from_pretrain(load_path)
    else:
        raise Exception('full_model_update must be chosen from lora finetune or full finetune')

    accelerator.print(f"use model in {load_path}")
    pairwise_rec.eval()

    pairwise_rec, cold_item_predict_dataloader= accelerator.prepare(
        pairwise_rec, pretrain_dataloader
    )
    #print(f"Model is on device: {next(pairwise_rec.parameters()).device}")
    item_list, predict_user_list = topk_predict(cold_item_predict_dataloader, pairwise_rec, accelerator)
    #test(cold_item_predict_dataloader, pairwise_rec,accelerator)

    accelerator.print("-----Prediction Finish-----")

    accelerator.print("-----Updating dataset-----")
    update_dataset(item_list, predict_user_list, accelerator)

    accelerator.print("-----Success-----")

def pretrain_loop(accelerator, review_data_loader, model, review_optimizer,
                     device, args, llm_weight_path, scheduler):
    review_best_loss = float('inf')

    for epoch in range(args.num_pretrained_epochs):
        review_total_loss = 0
        interaction_total_loss = 0
        # Initialize tqdm progress bar
        progress_bar = tqdm(review_data_loader, desc=f"Epoch {epoch + 1}",
                            disable=not accelerator.is_local_main_process, ncols=80)

        for item_id, users_id_tensor, prompt_ids, attention_mask in progress_bar:

            # Forward pass
            review_optimizer.zero_grad()

            outputs = model(item_id, prompt_ids, attention_mask, users_id_tensor, )
            review_loss = outputs[0]
            interaction_loss = outputs[1]
            if torch.isnan(review_loss):
                print(' ')
                continue

            # Backward pass and optimization
            #accelerator.backward(review_loss)
            #review_optimizer.step()
            #scheduler.step()

            review_total_loss += review_loss.item()
            interaction_total_loss += interaction_loss.item()
            # progress_bar.set_postfix({"Review Loss": review_loss.item()})
        thread_review_average_loss = torch.tensor([review_total_loss / len(review_data_loader)]).to(device)
        gathered_review_average_loss = accelerator.gather(thread_review_average_loss)
        review_average_loss = torch.mean(gathered_review_average_loss)

        interaction_average_loss = torch.tensor([interaction_total_loss / len(review_data_loader)]).to(device)
        gathered_interaction_average_loss = accelerator.gather(interaction_average_loss)
        interaction_average_loss = torch.mean(gathered_interaction_average_loss)

        accelerator.print(f"Epoch {epoch + 1} - Review Average Loss: {interaction_average_loss:.10f}")
        accelerator.print(f"Epoch {epoch + 1} - Review Average Loss: {review_average_loss:.10f}")

        # Check if the current loss is better than the best_loss
        accelerator.wait_for_everyone()
        if review_average_loss < review_best_loss:

            if args.full_model_update == 'lora finetune':
                model.eval()
                review_best_loss = review_average_loss
                lora_local_path = os.path.join(llm_weight_path, 'lora_finetune', f"{args.LLM_type}_{args.backbone_type}")
                user_emb_local_path = os.path.join(lora_local_path, 'user_embeddings.pt')
                mapper_local_path = os.path.join(lora_local_path, 'mapper.pt')
                # for name, param in accelerator.unwrap_model(model).named_parameters():
                #     if ("user_embeddings" in name) or ("mapper" in name) or ("lora" in name):
                #         accelerator.print(name,":",param)
                if accelerator.is_main_process:
                    accelerator.unwrap_model(model).encoder.llm.save_pretrained(lora_local_path)
                    torch.save(accelerator.unwrap_model(model).encoder.user_embeddings.state_dict(),
                               user_emb_local_path)
                    torch.save(accelerator.unwrap_model(model).encoder.mapper.state_dict(),
                               mapper_local_path)

                    accelerator.print(f'save weight in {lora_local_path}')
                else:
                    user_emb_local_path = os.path.join(llm_weight_path, 'full_finetune',
                                                       f"{args.LLM_type}_{args.backbone_type}")
                    accelerator.unwrap_model(model).encoder.llm.save_pretrained(user_emb_local_path)
                    mapper_local_path = os.path.join(user_emb_local_path, 'user_embeddings.pt')
                    torch.save(accelerator.unwrap_model(model).encoder.user_embeddings.state_dict(), mapper_local_path)




def topk_predict(cold_item_predict_dataloader, pairwise_rec, accelerator):
    with torch.no_grad():
        progress_bar = tqdm(cold_item_predict_dataloader,
                            disable=not accelerator.is_local_main_process, ncols=80)
        item_id_list = []
        predict_user = []
        total_lens = 0
        for item_id, users_id_tensor, prompt_ids, attention_mask in progress_bar:
            outputs = accelerator.unwrap_model(pairwise_rec).topk_predict(
                input_ids= prompt_ids, attention_mask=attention_mask
            )
            item_id_list.append(item_id)
            predict_user.append(outputs)
            total_lens += attention_mask.sum()
        avg_lens = total_lens / len(cold_item_predict_dataloader)
        print(avg_lens)
    return item_id_list, predict_user



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
        saved_file_path = os.path.join(dataset_root, f'{args.LLM_type}_predicted_cold_item_interaction_{args.lamda}.csv')

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
parser.add_argument('--pretrain_batch_size', type=int, default=32)
parser.add_argument('--finetune_batch_size', type=int, default=1)
parser.add_argument('--predict_batch_size',  type=int, default=8)

# backbone
parser.add_argument('--backbone_type', type=str, default='LightGCN')
parser.add_argument('--hidden_layers', type=list, default=[512, 256, 512], help='hidden_layers of mapper')

parser.add_argument('--num_pretrained_epochs', type=int, default=100) #如果使用多分类，10轮就足够了
parser.add_argument('--num_epochs', type=int, default=40)

parser.add_argument('--pretrain', type=int, default=0, help='0 for predict only, 1 for pretraining only, 2 for fine-tuning only, 3 for both')

parser.add_argument('--predict_way', type=str, default='positive', help='positive, negative, both')
parser.add_argument('--full_model_update', type=str, default='lora finetune', help='lora finetune or full finetune')
parser.add_argument('--lamda', type=int, default=10)


#TODO


args = parser.parse_args()

if __name__ == "__main__":
    main(args)
