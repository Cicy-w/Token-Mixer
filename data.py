import utils
import dgl
import torch
from ogb.nodeproppred import DglNodePropPredDataset
from ogb.linkproppred import DglLinkPropPredDataset
import scipy.sparse as sp
import os.path
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import  CoraFullDataset, AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset,CoauthorCSDataset,CoauthorPhysicsDataset
import random
import pickle as pkl
import numpy as np
from GraphDataset.make_dataset import get_train_val_test_split
from sklearn.preprocessing import StandardScaler
import GraphDataset.proj.functions as uf
from sklearn.model_selection import train_test_split
# from cache_sample import cache_sample_rand_csr

import torch_geometric.transforms as T
import os
import os.path as osp
from torch_geometric.datasets import Planetoid, Amazon, WikipediaNetwork, Actor, WikiCS, Coauthor
from torch_sparse import coalesce
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.utils.undirected import to_undirected
from torch_geometric.utils import remove_self_loops

from sklearn.model_selection import train_test_split
import pickle
# from dgl.data import ActorDataset

from sklearn.decomposition import PCA

class WebKB(InMemoryDataset):
    url = ('https://gitee.com/rockcor/geFom-gcn/tree/master/new_data')

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in ['cornell', 'texas', 'washington', 'wisconsin']

        super(WebKB, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for name in self.raw_file_names:
            download_url(f'{self.url}/{self.name}/{name}', self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')[1:-1]
            x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
            x = torch.tensor(x, dtype=torch.float)

            y = [int(r.split('\t')[2]) for r in data]
            y = torch.tensor(y, dtype=torch.long)

        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            edge_index, _ = remove_self_loops(edge_index)
            edge_index = to_undirected(edge_index)
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)

class dataset_heterophily(InMemoryDataset):
    def __init__(self, root='data/', name=None,
                 p2raw=None,
                 train_percent=0.01,
                 transform=None, 
                 pre_transform=None):
        if name=='actor':
            name='film'
        existing_dataset = ['chameleon', 'film', 'squirrel']
        if name not in existing_dataset:
            raise ValueError(
                f'name of hypergraph dataset must be one of: {existing_dataset}')
        else:
            self.name = name

        self._train_percent = train_percent

        if (p2raw is not None) and osp.isdir(p2raw):
            self.p2raw = p2raw
        elif p2raw is None:
            self.p2raw = None
        elif not osp.isdir(p2raw):
            raise ValueError(
                f'path to raw hypergraph dataset "{p2raw}" does not exist!')

        if not osp.isdir(root):
            os.makedirs(root)

        self.root = root

        super(dataset_heterophily, self).__init__(
            root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])
        self.train_percent = self.data.train_percent  #self.data.train_percent.item()

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        file_names = [self.name]
        return file_names

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        p2f = osp.join(self.raw_dir, self.name)
        with open(p2f, 'rb') as f:
            data = pickle.load(f)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)
def get_dataset(dataset, split_seed=0):

    if dataset in {"pubmed", "corafull", "photo", "cora", "citeseer"}:
        file_path = "GraphDataset/dataset/"+dataset+".pt"

        data_list = torch.load(file_path)

        adj = data_list[0]
        features = data_list[1]
        labels = data_list[2]

        idx_train = data_list[3]
        idx_val = data_list[4]
        idx_test = data_list[5]
        
        print(adj)
        if dataset == "pubmed":
            graph = PubmedGraphDataset()[0]
        elif dataset == "corafull":
            graph = CoraFullDataset()[0]
        # elif dataset == "computer":
        #     graph = AmazonCoBuyComputerDataset()[0]
        elif dataset == "photo":
            graph = AmazonCoBuyPhotoDataset()[0]
        elif dataset == "cs":
            graph = CoauthorCSDataset()[0]
        # elif dataset == "physics":
        #     graph = CoauthorPhysicsDataset()[0]
        elif dataset == "cora":
            graph = CoraGraphDataset()[0]
        elif dataset == "citeseer":
            graph = CiteseerGraphDataset()[0]

        graph = dgl.to_bidirected(graph)
        
        # LPE
        # lpe = utils.laplacian_positional_encoding(graph, 3) 
        # features = torch.cat((features, lpe), dim=1)
        # return adj, graph, features, labels, idx_train, idx_val, idx_test
    
    elif dataset in ['roman-empire', 'amazon_ratings', 'questions', 'minesweeper', 'tolokers']:
        data = np.load(os.path.join('data', f'{dataset.replace("-", "_")}.npz'))
        features = torch.tensor(data['node_features'])
        labels = torch.tensor(data['node_labels'])
        edges = torch.tensor(data['edges'])
        graph = dgl.graph((edges[:, 0], edges[:, 1]), num_nodes=len(features), idtype=torch.int)
        graph = dgl.to_bidirected(graph)
        train_masks = torch.tensor(data['train_masks'])
        val_masks = torch.tensor(data['val_masks'])
        test_masks = torch.tensor(data['test_masks'])

        split_idx = split_seed
        train_idx_list = [torch.where(train_mask)[0] for train_mask in train_masks]
        val_idx_list = [torch.where(val_mask)[0] for val_mask in val_masks]
        test_idx_list = [torch.where(test_mask)[0] for test_mask in test_masks]

        idx_train = train_idx_list[split_idx]
        idx_val = val_idx_list[split_idx]
        idx_test = test_idx_list[split_idx]
    
    elif dataset in ['actor']:
        
        split_idx = split_seed
        split_path = os.path.join('data', 'actor', 'raw', f'film_split_0.6_0.2_{split_idx}.npz')
        
        split_data = np.load(split_path)
        train_mask = split_data['train_mask']  # numpy bool array
        val_mask = split_data['val_mask']
        test_mask = split_data['test_mask']
        
        idx_train = torch.where(torch.tensor(train_mask))[0]
        idx_val = torch.where(torch.tensor(val_mask))[0]
        idx_test = torch.where(torch.tensor(test_mask))[0]
        
        total = len(idx_train) + len(idx_val) + len(idx_test)
        print(f"[Actor] Using pre-defined split {split_idx}:")
        print(f"  Train: {len(idx_train)}/{total} ({len(idx_train)/total*100:.1f}%)")
        print(f"  Val:   {len(idx_val)}/{total} ({len(idx_val)/total*100:.1f}%)")
        print(f"  Test:  {len(idx_test)}/{total} ({len(idx_test)/total*100:.1f}%)")
        
        data = Actor(root='./data/actor')
        pyg_graph = data[0]
        edge_index = pyg_graph.edge_index
        graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=pyg_graph.num_nodes)
        graph = dgl.to_bidirected(graph)
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)
        
        features = pyg_graph.x
        labels = pyg_graph.y
    

    elif dataset in['wisconsin', 'cornell', 'texas']:
        split_idx = split_seed
        split_path = os.path.join('data', dataset, 'raw', f'{dataset}_split_0.6_0.2_{split_idx}.npz')
    
        split_data = np.load(split_path)
        train_mask = split_data['train_mask']
        val_mask = split_data['val_mask']
        test_mask = split_data['test_mask']
        
        idx_train = torch.where(torch.tensor(train_mask))[0]
        idx_val = torch.where(torch.tensor(val_mask))[0]
        idx_test = torch.where(torch.tensor(test_mask))[0]
        
        total = len(idx_train) + len(idx_val) + len(idx_test)
        print(f"[{dataset.capitalize()}] Using pre-defined split {split_idx}:")
        print(f"  Train: {len(idx_train)}/{total} ({len(idx_train)/total*100:.1f}%)")
        print(f"  Val:   {len(idx_val)}/{total} ({len(idx_val)/total*100:.1f}%)")
        print(f"  Test:  {len(idx_test)}/{total} ({len(idx_test)/total*100:.1f}%)")
        
        name = dataset
        data = WebKB(root='./data/', name=name, transform=T.NormalizeFeatures())
        pyg_graph = data[0]
        edge_index = pyg_graph.edge_index
        graph = dgl.graph((edge_index[0], edge_index[1]))
        graph = dgl.to_bidirected(graph)
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)
        features = pyg_graph.x
        labels = pyg_graph.y
        
    elif dataset in ['chameleon', 'squirrel']:
        data_path = os.path.join('data', f'{dataset}.npz')
        data = np.load(data_path)
        features = torch.tensor(data['node_features'])
        labels = torch.tensor(data['node_labels'])
        edges = torch.tensor(data['edges'])
        graph = dgl.graph((edges[:, 0], edges[:, 1]), num_nodes=len(features), idtype=torch.int)
        graph = dgl.to_bidirected(graph)
        
        
    elif dataset in ['wikics']:
        root_path = './'
        path = osp.join(root_path, 'data', 'wikics')
        data = WikiCS(root=path, transform=T.NormalizeFeatures())
        graph = data[0]
        features = graph.x
        labels = graph.y
    elif dataset in ['computers']:
        root_path = './'
        path = osp.join(root_path, 'data', 'computers')
        data = Amazon(path, dataset, T.NormalizeFeatures())
        graph = data[0]
        features = graph.x
        labels = graph.y
    elif dataset in ['flickr']:
        train_percentage = 60
        load_default_split = train_percentage <= 0
        DATA_PATH = f'GraphDataset/dataset'
        adj_orig = pkl.load(open(f'{DATA_PATH}/{dataset}/{dataset}_adj.pkl', 'rb'))  # sparse
        features = pkl.load(open(f'{DATA_PATH}/{dataset}/{dataset}_features.pkl', 'rb'))  # sparase
        labels = pkl.load(open(f'{DATA_PATH}/{dataset}/{dataset}_labels.pkl', 'rb'))  # tensor
        if torch.is_tensor(labels):
            labels = labels.numpy()

        if load_default_split:
            tvt_nids = pkl.load(open(f'{DATA_PATH}/{dataset}/{dataset}_tvt_nids.pkl', 'rb'))  # 3 array
            train = tvt_nids[0]
            val = tvt_nids[1]
            test = tvt_nids[2]
        else:
            train, val, test = stratified_train_test_split(np.arange(len(labels)), labels, len(labels),
                                                           train_percentage)

        
        adj_orig = adj_orig.tocoo()
        U = adj_orig.row.tolist()
        V = adj_orig.col.tolist()
        g = dgl.graph((U, V))
        g = dgl.to_simple(g)
        g = dgl.remove_self_loop(g)
        graph = dgl.to_bidirected(g)

        if dataset in ['airport']:
            features = row_normalization(features)

        if sp.issparse(features):
            features = torch.FloatTensor(features.toarray())
        else:
            features = torch.FloatTensor(features)

        labels = torch.LongTensor(labels)
        idx_train = torch.LongTensor(train)
        idx_val = torch.LongTensor(val)
        idx_test = torch.LongTensor(test)

    elif dataset in ['dblp']:
        fname = f'GraphDataset/dataset/dblp/processed_dblp.pickle'
        if os.path.exists(fname):
            from torch_geometric.datasets import CitationFull
            # import torch_geometric.transforms as T
            data = CitationFull(root=f'./dataset', name=dataset, transform=T.NormalizeFeatures())[0]
            edges = data.edge_index
            features = data.x.numpy()
            labels = data.y.numpy()
            data_dict = {'edges': edges, 'features': features, 'labels': labels}
            uf.save_pickle(data_dict, fname)
        else:
            data_dict = uf.load_pickle(fname)
        edges, features, labels = data_dict['edges'], data_dict['features'], data_dict['labels']
        train, val, test = stratified_train_test_split(np.arange(len(labels)), labels, len(labels), 60)

        U = edges[0]
        V = edges[1]
        g = dgl.graph((U, V))
        g = dgl.to_simple(g)
        g = dgl.remove_self_loop(g)
        graph = dgl.to_bidirected(g)

        features = torch.FloatTensor(features)
        labels = torch.LongTensor(labels)
        idx_train = torch.LongTensor(train)
        idx_val = torch.LongTensor(val)
        idx_test = torch.LongTensor(test)
    elif dataset in ['physics','cs']:
        data = Coauthor(root='./data', name=dataset, transform=T.NormalizeFeatures())
        graph = data[0]
        features = graph.x
        labels = graph.y
    
    if dataset in ['physics', 'cs', 'wikics', 'computers']:
        edge_index = graph.edge_index
        graph = dgl.graph((edge_index[0], edge_index[1]))
        graph = dgl.to_simple(graph) ## gat2环境采用加这一行 gad的时候没有报错
        graph = dgl.to_bidirected(graph)
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)
    
    if dataset in ['chameleon', 'squirrel','physics','cs','wikics','computers']:
        random_state = np.random.RandomState(split_seed)
        n_samples = len(labels)
        indices = np.arange(n_samples)
        idx_train, idx_temp = train_test_split(indices, train_size=0.6, random_state=random_state, shuffle=True)
        idx_val, idx_test = train_test_split(idx_temp, train_size=0.5, random_state=random_state, shuffle=True)

        idx_train = torch.tensor(idx_train)
        idx_val = torch.tensor(idx_val)
        idx_test = torch.tensor(idx_test)
    if dataset not in ['actor', 'cornell', 'texas', 'wikics', 'wisconsin','computers']:
        lpe = utils.laplacian_positional_encoding(graph, 3) 
        features = torch.cat((features, lpe), dim=1)

    return graph, features, labels, idx_train, idx_val, idx_test


def col_normalize(mx):
    """Column-normalize sparse matrix"""
    scaler = StandardScaler()

    mx = scaler.fit_transform(mx)

    return mx

# def obtain_dataset(dataset, split_seed=0):
#     if dataset in ["cora", "pubmed", "citeseer", "computer", "photo"]:
#         G, adj, features, labels, idx_train, idx_val, idx_test = get_dataset(dataset, split_seed)
#     else:
#         G, features, nclass, idx_train, idx_val, idx_test, labels = preprocess_data(dataset, split_seed)
    
#     return G, features, idx_train, idx_val, idx_test, labels

# 分层采样
def stratified_train_test_split(label_idx, labels, n_nodes, train_rate, dataset=''):
    if dataset == 'cora':
        seed = 0
    else:
        seed = 2021
    n_train_nodes = int(train_rate / 100 * n_nodes)
    test_rate_in_labeled_nodes = (len(labels) - n_train_nodes) / len(labels)
    train_idx, test_and_valid_idx = train_test_split(
        label_idx, test_size=test_rate_in_labeled_nodes, random_state=seed, shuffle=True, stratify=labels)
    valid_idx, test_idx = train_test_split(
        test_and_valid_idx, test_size=.5, random_state=seed, shuffle=True, stratify=labels[test_and_valid_idx])
    return train_idx, valid_idx, test_idx

def row_normalization(mat):
    """Row-normalize sparse matrix"""
    row_sum = np.array(mat.sum(1))
    r_inv = np.power(row_sum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mat = r_mat_inv.dot(mat)
    return mat