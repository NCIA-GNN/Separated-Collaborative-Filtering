import torch
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
        # self.dropout_list = nn.ModuleList()
        self.dropout_list = dropout_list
        self.final_weight_dim = embedding_dim        
        for dim in self.weight_size:
            self.final_weight_dim+=dim
        
            
        self.model_list = nn.ModuleList()
        self.num_model = len(self.s_norm_adj_list) # clustered graph + full graph ex) 3 small cluster + 1 full graph = 4
        
        # Trial #2
        # self.user_local_embedding_aggregate = nn.Parameter(torch.zeros((self.incd_mat_list[0].shape[0], self.final_weight_dim),device='cuda',requires_grad = True))
        # self.item_local_embedding_aggregate = nn.Parameter(torch.zeros((self.incd_mat_list[0].shape[1], self.final_weight_dim),device='cuda',requires_grad = True))
        
        
        self.local_user_embeddings = nn.ModuleList()
        self.local_item_embeddings = nn.ModuleList()
        for i in range(self.num_model):
            n_users,n_items = self.incd_mat_list[i].shape[0], self.incd_mat_list[i].shape[1]
            self.model_list.append(NGCF(n_users, n_items, self.embedding_dim, self.weight_size, self.dropout_list))
            
            if i>=1:             
                # Trial #1
                # self.local_user_fc.append(nn.Linear(self.incd_mat_list[i].shape[0], self.incd_mat_list[0].shape[0]))
                # self.local_item_fc.append(nn.Linear(self.incd_mat_list[i].shape[1], self.incd_mat_list[0].shape[1]))
                
                # Trial #3
                # self.local_user_embeddings.append(nn.Embedding(self.incd_mat_list[i].shape[0], self.incd_mat_list[0].shape[0]))
                # self.local_item_embeddings.append(nn.Embedding(self.incd_mat_list[i].shape[1], self.incd_mat_list[0].shape[1]))
                self.local_user_embeddings.append(nn.Embedding(self.incd_mat_list[0].shape[0], self.incd_mat_list[i].shape[0]))
                self.local_item_embeddings.append(nn.Embedding(self.incd_mat_list[0].shape[1], self.incd_mat_list[i].shape[1]))
                
                
                
        self._init_weight_()
        
    def _init_weight_(self):
        for i in range(len(self.local_user_embeddings)):

            nn.init.xavier_uniform_(self.local_user_embeddings[i].weight)
            nn.init.xavier_uniform_(self.local_item_embeddings[i].weight)

    def forward(self, adj):
        user_embed_list = []
        item_embed_list = []
        
        for i in range(self.num_model):
            u_g_embeddings, i_g_embeddings = self.model_list[i](self.s_norm_adj_list[i])
            user_embed_list.append(u_g_embeddings)
            item_embed_list.append(i_g_embeddings)
            
         # full graph
        user_embd = user_embed_list[0]
        item_embd = item_embed_list[0]
        
        # Trial #3 : Similar to Trial #1, but use sparse matmul / RuntimeError: sparse_.is_sparse()INTERNAL ASSERT FAILED 
        # => if use 'torch.matmul' instead of 'torch.sparse.mm', it explodes.
        for i in range(1,self.num_model):
            
            ratio = float(item_embed_list[i].shape[0]/item_embd.shape[0])
            
            local_u_e = user_embed_list[i]
            local_i_e =  item_embed_list[i]

            u_e = torch.matmul(self.local_user_embeddings[i-1].weight,local_u_e)
            i_e = torch.matmul(self.local_item_embeddings[i-1].weight,local_i_e)
            
            ## 확인해보니 여기서 터짐
            user_embd = torch.add(user_embd, u_e, alpha = 1)
            item_embd = torch.add(item_embd, i_e, alpha = 1)
        
        
        
        # Trial #1 : local clustering embedding -> fully connected / memory overflow.. why? it works for 'ml-1m', not for 'gowalla','amazon-book'
        # for i in range(1,self.num_model):5
        #     ratio = float(item_embed_list[i].shape[0]/item_embd.shape[0])
        #     u_e = self.local_user_fc[i-1](torch.transpose(user_embed_list[i],0,1))
        #     i_e = self.local_item_fc[i-1](torch.transpose(item_embed_list[i],0,1))
        #     user_embd = torch.add(user_embd, u_e, alpha = ratio)
        #     item_embd = torch.add(item_embd, i_e, alpha = ratio)
        
        # Trial #2
        # I tried to assign local cluster embedding to full-size embedding / not working
        # for i in range(1,self.num_model):
        #     self.user_local_embedding_aggregate[self.idx_list[0][i-1],:]=user_embed_list[i]
        #     self.item_local_embedding_aggregate[self.idx_list[1][i-1],:]=item_embed_list[i]
        #     ratio = float(item_embed_list[i].shape[0]/origin_item_embd.shape[0])
        #     user_embd =  torch.add(user_embd ,self.user_local_embedding_aggregate, alpha = ratio)
        #     item_embd =  torch.add(item_embd ,self.item_local_embedding_aggregate, alpha = ratio)
            

        
        return user_embd, item_embd
  
    
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