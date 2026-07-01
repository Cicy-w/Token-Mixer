import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import os
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import random
from dgl import DGLGraph #, RandomWalkPE
import dgl

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def torch_sparse_tensor_to_sparse_mx(torch_sparse):
    """Convert a torch sparse tensor to a scipy sparse matrix."""

    m_index = torch_sparse._indices().numpy()
    row = m_index[0]
    col = m_index[1]
    data = torch_sparse._values().numpy()

    sp_matrix = sp.coo_matrix((data, (row, col)), shape=(torch_sparse.size()[0], torch_sparse.size()[1]))

    return sp_matrix

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def accuracy_batch(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds[:labels.shape[0]].eq(labels).double()
    correct = correct.sum()
    return correct

# add RWPE from dgl
# def randomwalk_positional_encoding(adj):
#     edge_index = adj._indices()
#     G = dgl.graph((edge_index[0], edge_index[1]))
#     transform = RandomWalkPE(k=5)
#     return transform(G)


# pos_enc_dim: position embedding size
def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """
    # Laplacian
    #adjacency_matrix(transpose, scipy_fmt="csr")
    # A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    A = g.adj().to_dense().numpy()
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with scipy
    #EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2) # for 40 PEs
    EigVec = EigVec[:, EigVal.argsort()] # increasing order
    lap_pos_enc = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 

    return lap_pos_enc

def normalize_adj(adj):
    """Row-column-normalize sparse matrix"""
    D = np.array(adj.sum(1))
    r_inv = np.power(D, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    adj = r_mat_inv.dot(adj).dot(r_mat_inv)
    return adj

def hop_features_gen(G, features, hop_num):
    
    adj = G.adj().to_dense()
    # adj = normalize_adj(adj)
    hop_features = torch.empty(features.shape[0], hop_num, features.shape[1])
    x = features + torch.zeros_like(features)
    for i in range(hop_num):
        x = torch.matmul(adj, x)
        for index in range(features.shape[0]):
            hop_features[index, i, :] = x[index]
    '''
    adj = G.adj().to_sparse() 
    
    hop_features = torch.empty(features.shape[0], hop_num, features.shape[1])
    x = features.clone()
    
    for i in range(hop_num):
        x = torch.sparse.mm(adj, x) 
        hop_features[:, i, :] = x
    '''
    return hop_features

def transition_probability_matrix(G, hop_num):
    """
    G: dgl graph
    """
    G = G.to('cuda')
    adj = G.adj().to_dense()
    degree = torch.sum(adj, dim=1)#.to_dense()
    degree_re = 1.0 / degree
    degree_re = degree_re.unsqueeze(1)
    # indices = torch.arrange(adj.size(0), device=adj.device).unsqueeze(0).repeat(2,1)
    
    tp_matrix = torch.mul(degree_re, adj)
    if hop_num > 1:
        hop_num = hop_num - 1
        for i in range(hop_num):
            tp_matrix = tp_matrix + torch.mul(degree_re, tp_matrix)
            
    for i in range(tp_matrix.shape[0]):
        tp_matrix[i][i] = 0
    sumRaw_tp = torch.sum(tp_matrix, dim=1)
    sumRaw_re = 1.0 / sumRaw_tp
    sumRaw_re = sumRaw_re.unsqueeze(1)
    tp_matrix = torch.mul(sumRaw_re, tp_matrix)
    
    return tp_matrix

def random_walk_with_length(G, start_node, length, path_num, isBack, seed=0, prob=None):
    """
    G: DGL graph
    """
    np.random.seed(seed)
    
    pathList = []
    for pathNum in range(path_num):
        path = []
        pre_node = None
        cur_node = int(start_node)
        path.append(cur_node)
        
        neighborsFirst = G.successors(start_node)
        if len(neighborsFirst)==0 or int(neighborsFirst[0])==-1:
            path.append(start_node)
            break
        
        for i in range(length):
            neighbors = G.successors(cur_node).tolist()
            lenNei = len(neighbors)
            
            if lenNei == 0:
                break
            elif lenNei == 1:
                target_node = neighbors[0]
            else:
                if isBack == False:
                    if pre_node != None and pre_node in neighbors:
                        neighbors.remove(pre_node)
                if prob == None:
                    target_node = np.random.choice(neighbors)
                else:
                    num_neighbors = len(neighbors)
                    p = np.empty(num_neighbors)
                    sum_p = 0
                    for jj in range(num_neighbors):
                        p[jj] = prob[cur_node][neighbors[jj]]
                        sum_p = sum_p + p[jj]
                    for jj in range(num_neighbors):
                        p[jj] = p[jj] / sum_p
                    # print(p)
                    if(len(p) == 1):
                        target_node = neighbors[0]
                    else:
                        target_node = np.random.choice(neighbors, p=p.ravel())
            path.append(int(target_node))
            pre_node = cur_node
            cur_node = target_node
        pathList.append(path)
    
    return pathList


# from transformers import BertModel

def mixed_walk_gen(G, t_num, w_len, dataset, seed, uniformRWRate, nonBackRWRate, nJumpRate):
    # nodes_features = torch.empty(G.nodes().size(dim=0), t_num*w_len)
    tp_matrix = transition_probability_matrix(G, 3)

    nodes_paths = [] 
    rWNum = int(t_num * uniformRWRate)
    nBRWNum = int(t_num * nonBackRWRate)
    nJWNum = int(t_num * nJumpRate)
    nBNJWNum = t_num - rWNum - nBRWNum - nJWNum
    for i in range(len(G.nodes())):
        random_walk_pathList = random_walk_with_length(G=G, start_node=i, length=w_len, path_num=rWNum, isBack=True, seed=seed, prob=None)
        non_backtracking_pathList = random_walk_with_length(G=G, start_node=i, length=w_len, path_num=nBRWNum, isBack=False, seed=seed+1, prob=None)
        neighborhood_jump_pathList = random_walk_with_length(G=G, start_node=i, length=w_len, path_num=nJWNum, isBack=True, seed=seed+2, prob=tp_matrix)
        non_backtracking_neighborhood_jump_pathList = random_walk_with_length(G=G, start_node=i, length=w_len, path_num=nBNJWNum, isBack=False, seed=seed+3, prob=tp_matrix)
        pathList = random_walk_pathList + non_backtracking_pathList + neighborhood_jump_pathList + non_backtracking_neighborhood_jump_pathList
        nodes_paths.append(pathList)
    if not os.path.exists("./DatasetPathInfo/" + dataset):
        os.makedirs("./DatasetPathInfo/" + dataset)
    torch.save(nodes_paths, "./DatasetPathInfo/" + dataset + "/" + dataset + '_num=' + str(t_num) + '_len=' + str(w_len) + '_uniformRWRate=' + str(uniformRWRate) + '_nonBackRWRate=' + str(nonBackRWRate) + '_nJumpRate=' + str(nJumpRate) + '.pt')

def get_token(args, G, features, W, num_steps, dataset, globalToken, hopNum, uniformRWRate, nonBackRWRate, nJumpRate):
    if globalToken == True:
        cache_dir = f"./cache/{args.dataset}/alpha_{args.alpha}"
        cache_path = f"{cache_dir}/global_token_topk{args.topk}.pt"
        
        nodes_features = torch.empty(features.shape[0], W+1+1+hopNum, features.shape[1]*2)
        print("generating node global embedding...")
        if os.path.exists(cache_path):
            global_token = torch.load(cache_path)
        else:
            os.makedirs(cache_dir, exist_ok=True)

            sim_cn = similarity_CN(G, features, args.alpha)
            global_token = global_cn(sim_cn, features, args.topk)
            torch.save(global_token, cache_path)
    else:
        nodes_features = torch.empty(features.shape[0], W+1 +hopNum, features.shape[1]*2) # normal
    
    print('loading stored path...')
    pt = torch.load("./DatasetPathInfo/" + dataset + "/" + dataset + '_num=' + str(W) + '_len=' + str(num_steps) + '_uniformRWRate=' + str(uniformRWRate) + '_nonBackRWRate=' + str(nonBackRWRate) + '_nJumpRate=' + str(nJumpRate) + '.pt', map_location=torch.device('cpu'))
    print("done")
    
    print('generating tokens for each node...')
    # disNum = 3
    # if not os.path.exists("./DatasetDistanceInfo/" + dataset):
    #     os.makedirs("./DatasetDistanceInfo/" + dataset)
    
    # hopNum = 3
    if not os.path.exists("./DatasetHopInfo/" + dataset):
        os.makedirs("./DatasetHopInfo/" + dataset)
    
    # hop features
    if os.path.exists("./DatasetHopInfo/" + dataset + "/" + dataset + ".pt"):
        hop_features = torch.load("./DatasetHopInfo/" + dataset + "/" + dataset + "_hop" + str(hopNum) + ".pt")
    else:
        hop_features = hop_features_gen(G, features, hopNum)
        torch.save(hop_features, "./DatasetHopInfo/" + dataset + "/" + dataset + "_hop" + str(hopNum)  + ".pt")
    
    for node in range(features.shape[0]):
        i = 0
        feature_raw = features[node] 
        
        # ------------------------- self-token -------------------------
        feature = torch.cat([feature_raw, feature_raw], dim=0)
        nodes_features[node, i, :] = feature  
        i += 1
        
        if globalToken == True:
            global_feature = torch.cat([feature_raw, global_token[node]], dim=0)
            nodes_features[node, i, :] = global_feature
            i += 1
        # ------------------------- hop-token -------------------------
        # hop_features = hop_features_gen(G, features, hopNum)
        for hop_emb in hop_features[node]:
            feature = torch.cat([feature_raw, hop_emb], dim=0)
            nodes_features[node, i, :] = feature  
            i += 1

        # ------------------------- path-token -------------------------
        walk = pt[node]
        for path in walk:
            feature = torch.zeros(features.shape[1])

            isFirstNode = True
            for node2 in path:
                if isFirstNode == True:
                    isFirstNode = False
                    continue
                nf = features[node2]
                feature = feature + nf 
        
            feature = torch.cat([feature_raw, feature], dim=0)
            nodes_features[node, i, :] = feature  
            i += 1
            
    print("done")
    print(nodes_features.size())      
    return nodes_features

from sklearn.metrics.pairwise import cosine_similarity as cos
def similarity_CN(data, feature, alpha):
    edge_index = torch.stack(data.edges()) # 2,E

    self_loop_mask = (edge_index[0] != edge_index[1])
    edge_index = edge_index[:, self_loop_mask]
    
    adj = edge_index_to_adj_sparse(edge_index)
    Scn = torch.mm(adj,adj)

    fea_dist = cos(feature.numpy()) 
    fea_dist = torch.from_numpy(fea_dist)

    sim = alpha*Scn + (1-alpha)*fea_dist

    return sim

def global_cn(sim_cn, features,topk):
    N = features.shape[0]
    global_tokens = torch.zeros(N, features.shape[1], device=features.device)
    
    for i in range(N):
        sim_row = sim_cn[i]
        
        sim_row_no_self = sim_row.clone()
        sim_row_no_self[i] = -float('inf')
        
        top_k_values, top_k_indices = torch.topk(sim_row_no_self, topk)
        weights = F.softmax(top_k_values, dim=0)
        
        top_k_features = features[top_k_indices]
        global_tokens[i] = (weights.unsqueeze(1)*top_k_features).sum(dim=0)
        
        # global_tokens[i] = top_k_features.sum(dim=0)
    
    return global_tokens
    
def edge_index_to_adj_sparse(edge_index, num_nodes=None):
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1
    
    E = edge_index.size(1)
    values = torch.ones(E, device=edge_index.device)
    
    adj = torch.sparse_coo_tensor(
        indices=edge_index,
        values=values,
        size=(num_nodes, num_nodes)
    )
    
    adj = adj + adj.t()
    
    adj_dense = adj.to_dense()
    adj_dense.fill_diagonal_(0)
    adj_dense = (adj_dense > 0).float()
    
    return adj_dense
    
