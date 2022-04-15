
import torch.nn as nn
import torch.sparse as sparse
import torch.nn.functional as F
import numpy as np
from main import *
    
#Universial Clustering Recommendation
class UCR(nn.Module): 
    
    def __init__(self, s_norm_adj_list, incd_mat_list , idx_list , alg_type, embedding_dim, weight_size, dropout_list):
        super().__init__()
        self.s_norm_adj_list = s_norm_adj_list
        self.incd_mat_list = incd_mat_list
        
        self.idx_list = idx_list
        self.alg_type = alg_type
        self.embedding_dim = embedding_dim
        self.weight_size = weight_size
        self.n_layers = len(self.weight_size)
        self.dropout_list = dropout_list
        self.final_weight_dim = embedding_dim        
        for dim in self.weight_size:
            self.final_weight_dim+=dim
        
            
        self.model_list = nn.ModuleList()
        self.num_model = len(self.s_norm_adj_list) # clustered graph + full graph ex) 3 small cluster + 1 full graph = 4
        self.local_user_embeddings = []
        self.local_item_embeddings = []
    
        for i in range(self.num_model):
            n_users,n_items = self.incd_mat_list[i].shape[0], self.incd_mat_list[i].shape[1]
            if self.alg_type in ['ngcf' ,'NGCF']:
                self.model_list.append(NGCF(n_users, n_items, self.embedding_dim, self.weight_size, self.dropout_list))
            elif self.alg_type in ['mf' ,'MF']:
                
                self.model_list.append(MF(n_users, n_items, self.embedding_dim))
                self.final_weight_dim = self.embedding_dim
                
            elif self.alg_type in ['lightgcn' ,'LightGCN']:
                self.model_list.append(LightGCN(n_users, n_items, self.embedding_dim, self.weight_size, self.dropout_list))
            
            if i>=1:             
                with torch.no_grad():
                    self.local_user_embeddings.append(torch.zeros((self.incd_mat_list[0].shape[0], self.final_weight_dim),requires_grad = True,device='cuda').cuda())
                    self.local_item_embeddings.append(torch.zeros((self.incd_mat_list[0].shape[1], self.final_weight_dim),requires_grad = True,device='cuda').cuda())
        
        self.W_ratio_u = nn.Embedding(self.incd_mat_list[0].shape[0], self.final_weight_dim)
        self.W_ratio_i = nn.Embedding(self.incd_mat_list[0].shape[1], self.final_weight_dim)
        
        self._init_weight_()
    def _init_weight_(self):
        nn.init.xavier_uniform_(self.W_ratio_u.weight)
        nn.init.xavier_uniform_(self.W_ratio_i.weight)

    def forward(self, s_norm_adj_list):
        
        user_embed_list = []
        item_embed_list = []
        for i in range(self.num_model):
            u_g_embeddings, i_g_embeddings = self.model_list[i](s_norm_adj_list[i])
            user_embed_list.append(u_g_embeddings)
            item_embed_list.append(i_g_embeddings)
         # full graph
        user_embd = user_embed_list[0]
        item_embd = item_embed_list[0]

        with torch.no_grad():
            for i in range(1,self.num_model):
                self.local_user_embeddings[i-1][self.idx_list[0][i-1]]=user_embed_list[i]
                self.local_item_embeddings[i-1][self.idx_list[1][i-1]]=item_embed_list[i]
        local_user_embd = torch.sum(torch.stack(self.local_user_embeddings, dim=2),dim=2)
        local_item_embd = torch.sum(torch.stack(self.local_item_embeddings, dim=2),dim=2)
        final_u = user_embd + torch.mul(self.W_ratio_u.weight , local_user_embd)
        final_i = item_embd + torch.mul(self.W_ratio_i.weight , local_item_embd)
        
        return final_u, final_i
        
  
    
class NGCF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, weight_size, dropout_list):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.weight_size = weight_size
        self.n_layers = len(self.weight_size)
        self.dropout_list = nn.ModuleList()
        self.GC_Linear_list = nn.ModuleList()
        self.Bi_Linear_list = nn.ModuleList()

        self.weight_size = [self.embedding_dim] + self.weight_size
        for i in range(self.n_layers):
            self.GC_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i+1]))
            self.Bi_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i+1]))
            self.dropout_list.append(nn.Dropout(dropout_list[i]))

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        self._init_weight_()
        
    def _init_weight_(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, adj):
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        
        for i in range(self.n_layers):
            
            side_embeddings = torch.matmul(adj, ego_embeddings)
            # side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            
            sum_embeddings = F.leaky_relu(self.GC_Linear_list[i](side_embeddings))

            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            
            bi_embeddings = F.leaky_relu(self.Bi_Linear_list[i](bi_embeddings))
            ego_embeddings = sum_embeddings + bi_embeddings
            ego_embeddings = self.dropout_list[i](ego_embeddings)

            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings += [norm_embeddings]

        all_embeddings = torch.cat(all_embeddings, dim=1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        
        return u_g_embeddings, i_g_embeddings
    
class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, weight_size, dropout_list):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.weight_size = weight_size
        self.n_layers = len(self.weight_size)
        self.dropout_list = nn.ModuleList()
        self.GC_Linear_list = nn.ModuleList()
        self.Bi_Linear_list = nn.ModuleList()

        self.weight_size = [self.embedding_dim] + self.weight_size
        for i in range(self.n_layers):
            self.GC_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i+1]))
            self.Bi_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i+1]))
            self.dropout_list.append(nn.Dropout(dropout_list[i]))

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        self._init_weight_()
        
    def _init_weight_(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, adj):
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        '''
        Not Yet implemented
        '''
        for i in range(self.n_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            sum_embeddings = F.leaky_relu(self.GC_Linear_list[i](side_embeddings))
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bi_embeddings = F.leaky_relu(self.Bi_Linear_list[i](bi_embeddings))
            ego_embeddings = sum_embeddings + bi_embeddings
            ego_embeddings = self.dropout_list[i](ego_embeddings)

            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings += [norm_embeddings]

        all_embeddings = torch.cat(all_embeddings, dim=1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        
        return u_g_embeddings, i_g_embeddings

class MF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        self._init_weight_()

    def _init_weight_(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, adj):
        u_g_embeddings = self.user_embedding.weight
        i_g_embeddings = self.item_embedding.weight
        return u_g_embeddings, i_g_embeddings


class GCMC(nn.Module):
    def __init__(self):
        # TODO
        pass
    def _init_weight_(self):
        # TODO
        pass
    def forward(self):
        # TODO
        pass


class GCN(nn.Module):
    def __init__(self):
        # TODO
        pass
    def _init_weight_(self):
        # TODO
        pass
    def forward(self):
        # TODO
        pass

# def   sparse_mx_to_torch_sparse_tensor(sparse_mx):
#     """Convert a scipy sparse matrix to a torch sparse tensor."""
#     sparse_mx = sparse_mx.tocoo().astype(np.float32)
#     indices = torch.from_numpy(
#         np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
#     values = torch.from_numpy(sparse_mx.data)
#     shape = torch.Size(sparse_mx.shape)
#     return torch.sparse.FloatTensor(indices, values, shape)