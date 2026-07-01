import time
from GraphDataset.data import get_dataset
import utils
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from early_stop import EarlyStopping, Stop_args
from model import MLPModel #TransformerModel,,MLPModel_NodeWise
from lr import PolynomialDecayLR
import os.path
import torch.utils.data as Data
import argparse
import networkx as nx
import pandas as pd
from sklearn.metrics import roc_auc_score

# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
# torch.use_deterministic_algorithms(True)

from plot import *
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser()
    # main parameters
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='cora',
                        help='Choose from {pubmed, cora, citeseer, photo, cornell, chameleon, squirrel, actor, wisconsin, wikics, computers, physics, cs, texas, tolokers}')
    parser.add_argument('--device', type=int, default=0, help='Device cuda id')
    parser.add_argument('--seed', type=int, default=3407, help='Random seed.')

    # model parameters
    parser.add_argument('--t_nums', type=int, default=20, help='nums of token_paths')
    parser.add_argument('--w_len', type=int, default=20, help='max walk_length of token_path')
    parser.add_argument('--pe_dim', type=int, default=15, help='position embedding size')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden layer size')
    parser.add_argument('--ffn_dim', type=int, default=64, help='FFN layer size')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of Transformer layers')
    parser.add_argument('--n_heads', type=int, default=1, help='Number of Transformer heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout')
    parser.add_argument('--attention_dropout', type=float, default=0.6, help='Dropout in the attention layer')

    # training parameters
    parser.add_argument('--batch_size', type=int, default=8000, help='Batch size')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
    parser.add_argument('--tot_updates', type=int, default=2000, help='used for optimizer learning rate scheduling')
    parser.add_argument('--warmup_updates', type=int, default=400, help='warmup steps')
    parser.add_argument('--peak_lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--end_lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--uniformRWRate', type=float, default=0.25, help='rate for uniform random walk')
    parser.add_argument('--nonBackRWRate', type=float, default=0.25, help='rate for non-backtracking random walk')
    parser.add_argument('--nJumpRate', type=float, default=0.25, help='rate for neighborhood jump walk')
    parser.add_argument('--patience', type=int, default=50, help='Patience for early stopping')
    parser.add_argument('--global_token',  type=bool, default=True, help='use global token or not')
    parser.add_argument('--hop_num', type=int, default=3, help='number of hop token')
    # parser.add_argument('--k1', type=int, default=24)
    parser.add_argument('--model', type=str, default='MLP', help='choice in [transformer, MLP, Polytransformer, ablation]')
    parser.add_argument('--activation', type=str, default='softmax', help='choice in [tanh, relu, softmax]')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--belta', type=float, default=0.7)
    return parser.parse_args()

def run_single_experiment(run_idx, args, device):
    seed = args.seed + run_idx
    set_seed(seed)
    
     
    if torch.cuda.is_available():
        torch.cuda.init() 
        
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated(device)/1024/1024
    
    if args.dataset in ['tolokers', 'minesweeper', 'actor', 'wisconsin', 'cornell', 'texas']:
        G, features, labels, idx_train, idx_val, idx_test = get_dataset(args.dataset, split_seed=run_idx)
    else:
        G, features, labels, idx_train, idx_val, idx_test = get_dataset(args.dataset, split_seed=seed)
    
    
    is_binary = (labels.max().item() == 1)
    
    
    if is_binary:
        labels = labels.float()

    path = f"./DatasetPathInfo/{args.dataset}/{args.dataset}_num={args.t_nums}_len={args.w_len}_uniformRWRate={args.uniformRWRate}_nonBackRWRate={args.nonBackRWRate}_nJumpRate={args.nJumpRate}.pt"
    if not os.path.isfile(path):
        utils.mixed_walk_gen(G, args.t_nums, args.w_len, args.dataset, seed=42,
                             uniformRWRate=args.uniformRWRate,
                             nonBackRWRate=args.nonBackRWRate,
                             nJumpRate=args.nJumpRate)

    processed_features = utils.get_token(args,G, features, args.t_nums, args.w_len, args.dataset,
                                         args.global_token, args.hop_num,
                                         args.uniformRWRate, args.nonBackRWRate,
                                         args.nJumpRate).to(device)
    # d = processed_features.shape[-1] // 2
    # processed_features = processed_features[:,:,:d]
    
    labels = labels.to(device)

   
    batch_data_train = Data.TensorDataset(processed_features[idx_train], labels[idx_train])
    batch_data_val   = Data.TensorDataset(processed_features[idx_val],   labels[idx_val])
    batch_data_test  = Data.TensorDataset(processed_features[idx_test],  labels[idx_test])

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = Data.DataLoader(batch_data_train, batch_size=args.batch_size, shuffle=True, generator=g)
    val_loader   = Data.DataLoader(batch_data_val,   batch_size=args.batch_size, shuffle=False)
    test_loader  = Data.DataLoader(batch_data_test,  batch_size=args.batch_size, shuffle=False)

    
    if args.global_token:
        numTokens = args.t_nums + 1 + 1 + args.hop_num
    else:
        numTokens = args.t_nums + 1 + args.hop_num
    
    numnodes = features.shape[0]
    
    num_classes = 2 if is_binary else labels.max().item() + 1
    
    if args.model == 'transformer':
        model = TransformerModel(t_nums=numTokens,
                                 n_class=num_classes,
                                 input_dim=features.shape[1] * 2,
                                 pe_dim=args.pe_dim,
                                 n_layers=args.n_layers,
                                 num_heads=args.n_heads,
                                 hidden_dim=args.hidden_dim,
                                 ffn_dim=args.ffn_dim,
                                 dropout_rate=args.dropout,
                                 attention_dropout_rate=args.attention_dropout).to(device)
    elif args.model == 'MLP':
        model = MLPModel(args, k_token=numTokens, input_dim=processed_features.shape[-1],
                         hidden_dim=args.hidden_dim, num_classes=num_classes,
                         dropout=args.dropout, k1=numTokens).to(device)
        
        # from thop import profile, clever_format
        # dummy_input = torch.randn(1, numTokens, processed_features.shape[-1]).to(device)
        # flops, params = profile(model, inputs=(dummy_input, ), verbose=False)
        # macs = flops /2 # flops->macs
        # macs, params = clever_format([macs, params], "%.3f")
        # print(f"Model MACs: {macs}, Params:{params}")
        
    else:
        raise ValueError(f"Unknown model: {args.model}")

   
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialDecayLR(optimizer,
                                     warmup_updates=args.warmup_updates,
                                     tot_updates=args.tot_updates,
                                     lr=args.peak_lr,
                                     end_lr=args.end_lr,
                                     power=1.0)

    from early_stop import StopVariable, Best
    if is_binary:
        stopping_args = dict(
            stop_varnames=[StopVariable.AUROC, StopVariable.LOSS],
            patience=args.patience,
            max_epochs=args.epochs,
            remember=Best.RANKED   
        )
    else:
        stopping_args = dict(
            stop_varnames=[StopVariable.ACCURACY, StopVariable.LOSS],
            patience=args.patience,
            max_epochs=args.epochs,
            remember=Best.RANKED   
        )

    early_stopping = EarlyStopping(model, **stopping_args)
    
    best_val_metric = -float('inf')
    best_state = None
    
    train_time = []

    for epoch in range(args.epochs):
        torch.cuda.synchronize(device)
        epoch_start = time.time()
        model.train()
        loss_train_sum = 0.0
        acc_train_sum = 0.0
        
        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            if args.model  == 'ablation':
                out = model(x, idx_train)
            else:
                out = model(x)
            
            if is_binary:
                loss = F.nll_loss(out, y.long())
            else:
                loss = F.nll_loss(out, y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            
            loss_train_sum += loss.item()
            if is_binary:
                acc_train = utils.accuracy_batch(out, y.long())
            else:
                acc_train = utils.accuracy_batch(out, y)
            acc_train_sum += acc_train.item()
        
        torch.cuda.synchronize(device)    
        epoch_train_time = time.time() - epoch_start
        train_time.append(epoch_train_time)

       
        model.eval()
        loss_val_sum = 0.0
        acc_val_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                if args.model  == 'ablation':
                    out = model(x, idx_val)
                else:
                    out = model(x)
                loss_val_sum += F.nll_loss(out, y).item()
                acc_val_sum += utils.accuracy_batch(out, y).item()
        val_metric = acc_val_sum / len(idx_val)
        metric_name = "Acc"
        
        avg_loss_val = loss_val_sum / len(val_loader)
        
        if is_binary:
            print(f'Epoch {epoch+1:04d} | Train Loss: {loss_train_sum:.4f} | Val Loss: {avg_loss_val:.4f} | Val AUROC: {val_metric:.4f}')
        else:
            print(f'Epoch {epoch+1:04d} | Train Loss: {loss_train_sum:.4f} | Val Loss: {avg_loss_val:.4f} | Val Acc: {val_metric:.4f}')
        
        
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        
        if early_stopping.check([val_metric, avg_loss_val], epoch):
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            test_labels = y
            if args.model  == 'ablation':
                out = model(x, idx_test)
            else:
                out = model(x)
            test_loss += F.nll_loss(out, y).item()
            test_acc += utils.accuracy_batch(out, y).item()
    test_metric = test_acc / len(idx_test)
    metric_name = "Accuracy"
    
    test_loss /= len(test_loader)
    print(f"\nTest {metric_name}: {test_metric:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    torch.cuda.synchronize(device)
    peak_memory = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    mem_after = torch.cuda.memory_allocated(device) / 1024 / 1024
    
    
    # plot_heatmap(attn_matrix, args.dataset)
    
    if len(train_time)>1:
        avg_train_time = np.mean(train_time[1:])
    else:
        avg_train_time = train_time[0] if len(train_time) == 1 else 0
    
    return test_loss, test_metric, avg_train_time, peak_memory, mem_before, mem_after#, macs, params

def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    RUNS = args.runs
    losses = []
    metrics = []
    
    train_time_runs = []
    for run_idx in range(RUNS):
        print(f"\n{'='*60}")
        print(f"Running {run_idx + 1}/{RUNS}")
        print(f"{'='*60}")
        loss, metric, avg_train_time, peak_memory, mem_before,mem_after = run_single_experiment(run_idx, args, device) #, macs, params
        print(f"Run {run_idx + 1} -> Test Loss: {loss:.4f}, Test Metric: {metric:.4f}")
        losses.append(loss)
        metrics.append(metric)
        train_time_runs.append(avg_train_time)
    avg_train_time = np.mean(train_time_runs)
    
    mean_loss = np.mean(losses)
    std_loss = np.std(losses, ddof=1)
    mean_metric = np.mean(metrics)
    std_metric = np.std(metrics, ddof=1)

    is_binary = (args.dataset == 'tolokers' or args.dataset == 'minesweeper')
    metric_name = "AUROC" if is_binary else "Accuracy"

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Test Loss: {mean_loss:.4f} ± {std_loss:.4f}")
    print(f"Test {metric_name}: {mean_metric:.4f} ± {std_metric:.4f}")

    print(f"Average training time:{avg_train_time}")
    print(f"Peak GPU memory: {peak_memory:.2f}MB")
    print(f"Memory before: {mem_before:.2f}MB")
    print(f"Memory after: {mem_after:.2f}MB")
    consum_mem = mem_after-mem_before
    print(f"Memory consume: {consum_mem:.2f}MB")

    result_row = {
        'dataset': args.dataset,
        'seed':args.seed,
        'model': args.model,
        'global-token':args.global_token,
        'topk': args.topk,
        "alpha":args.alpha,
        'lr': args.peak_lr,
        'weight_decay': args.weight_decay,
        'hidden': args.hidden_dim,
        'ffn_dim': args.ffn_dim,
        'dropout': args.dropout,
        'attn_drop': args.attention_dropout, 
        'hop_num': args.hop_num,
        't_nums': args.t_nums,
        'w_len': args.w_len,
        'urw': args.uniformRWRate, 
        'nbrw': args.nonBackRWRate, 
        'njw': args.nJumpRate,
        'runs': args.runs,
        'patience': args.patience,
        f'test_{metric_name.lower()}': f"{mean_metric:.4f} ± {std_metric:.4f}", 
        # 'epoch_train_time':  f"{avg_train_time}s",
        # 'peak_gpu': f"{peak_memory}MB",
        # 'MACs': f"{macs}",
        # 'parameters':f"{params}"
    }
    df = pd.DataFrame([result_row])
    # csv_file = f"{args.dataset}_{args.model}_summary.csv"
    csv_file = f"token_wise_mlp_{args.dataset}_results.csv"
    # csv_file = f"ablation_results.csv"
    # csv_file = f"efficient_results.csv"
    if os.path.isfile(csv_file):
        df.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_file, mode='w', header=True, index=False)
    print(f"\nSummary results saved to {csv_file}")

if __name__ == '__main__':
    main()
