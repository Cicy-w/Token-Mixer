
Token-Mixer: An MLP-Like Architecture for Tokenized Graph Learning

## Environment Settings    
- pytorch 2.0.0+cu118
- numpy 1.21.5
- torch-geometric 2.3.0
- tqdm 4.67.1
- scipy 1.8.0
- seaborn 0.13.2


### Running the code
You can run the following Command:
'''sh
python train.py --dataset cora --peak_lr 0.005 --weight_decay 1e-5 --hidden_dim 256 --ffn_dim 64 --dropout 0.5 --attention_dropout 0.4 --t_nums 40 --w_len 4 \
  --uniformRWRate 0.3 --nonBackRWRate 0.05 --nJumpRate 0.6 --patience 50 --runs 1 --alpha 0.3 --topk 1  
'''


