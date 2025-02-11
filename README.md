# LLM-InS
This is the code for our paper "Large Language Model Interaction Simulator for Cold-Start Item Recommendation".

# Structure
## model 
- our model of the paper LLM-InS
  - filtering: include SubTower model
  - refining: include prompt construct and ask LLM for the answer
  - updating: include recommendation model train
 
## Environment Preparation Before Running

- **GPU:** NVIDIA A800
- **Dependencies from GitHub:**
  - `Llama2-7b`
  - `Llama-Factory`
  - `Chinese-LLaMA-Alpaca`
  - Official libraries for models like `LightGCN`, etc.
- **Data Processing:**
  - Division of cold and hot data
    - LLM-SFT
- Follow the tutorial to:
  - Download model parameters for Llama2 (the paper uses parameters model for llama7b)
  - Construct SFT training data based on different datasets
    - Execute command XXX to generate corresponding SFT training set for cold items for training
    - Perform parameter merge
    - Save the model

### Refining
- `Llama-SubTower`
- `CollaborativeTower`

### Filtering
- Construct datasets for cold items
- Refine interaction with prompts
- Obtain the final interaction CSV file

### Updating
- Following models like LightGCN, NGCF, etc., shuffle generated interactions with existing interactions for training
- Obtain the final recommendation cold start embedding.
