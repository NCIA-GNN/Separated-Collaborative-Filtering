'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
from sklearn.cluster import SpectralCoclustering

class Data(object):
    def __init__(self, path, batch_size, spectral_cc=False, create=False):
        self.path = path
        self.batch_size = batch_size
        self.spectral_cc = spectral_cc
        self.train_file = path + '/train.txt'
        self.test_file = path + '/test.txt'
        self.create = create
        #get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}

        self.exist_users = []

        with open(self.train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)

        with open(self.test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)
        self.n_items += 1
        self.n_users += 1

        
           
        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.row, self.col = self.R.shape[0], self.R.shape[1]
        print("Original R size", self.R.shape)
        self.R_Item_Interacts = sp.dok_matrix((self.n_items, self.n_items), dtype=np.float32)

        self.train_items, self.test_set = {}, {}
        with open(self.train_file) as f_train:
            with open(self.test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:]

                    for idx, i in enumerate(train_items):
                        self.R[uid, i] = 1.

                    self.train_items[uid] = train_items

                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue

                    uid, test_items = items[0], items[1:]
                    self.test_set[uid] = test_items

    def get_adj_mat(self, scc=0, create=0, N=5, cl_num=0):
        t1 = time()
        if create ==1:
            adj_mat, norm_adj_mat, mean_adj_mat, R = self.create_adj_mat()
            sp.save_npz(self.path + '/s_incd_mat.npz', R)
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)      

        if scc==0 : 
            
            # Load Incidence Matrix
            R = sp.load_npz(self.path + '/s_incd_mat.npz')
            
            # Clustering
#             N = 5
            
            bicl = SpectralCoclustering(n_clusters=N, random_state=0)
            bicl.fit(R)
            print('Row (user) cluster counts:', np.unique(bicl.row_labels_, return_counts=True)[1])
            print('Column (Item) cluster counts:', np.unique(bicl.column_labels_, return_counts=True)[1])
            
            # Select first cluster
            sample_cluster_num = cl_num
            row_idx, col_idx = np.argsort(bicl.row_labels_), np.argsort(bicl.column_labels_)
            sorted_row = sorted(bicl.row_labels_)
            chenge_points_row= []
            for i in range(len(sorted_row)-1):
                if i ==0:
                    chenge_points_row.append(i)
                if sorted_row[i]!=sorted_row[i+1]:
                    chenge_points_row.append(i+1)
            
            # Get new incidence matrix from user listbelongs to first cluster (ignore clustered items)
            if sample_cluster_num == (N-1) : 
                sample_row_idx = sorted(row_idx[chenge_points_row[sample_cluster_num]:]) 
            else : 
                sample_row_idx = sorted(row_idx[chenge_points_row[sample_cluster_num]:chenge_points_row[sample_cluster_num+1]])
            item_list = []            
            for u in sample_row_idx:
                for i in self.train_items[u]:
                    item_list.append(i)
                for i in self.test_set[u]:
                    item_list.append(i)
            sample_col_idx = sorted(list(set(item_list)))
            R = R[sample_row_idx,:]
            R = R[:,sample_col_idx]            


            # filtering clustered items
            sampled_train_set = {k:v for k,v in self.train_items.items() if k in sample_row_idx}
            sampled_test_set = {k:v for k,v in self.test_set.items() if k in sample_row_idx}
            
            # re-index the sampled items
            user_index = dict(zip(sample_row_idx,range(len(sample_row_idx))))
            item_index = dict(zip(sample_col_idx,range(len(sample_col_idx))))         
            new_train_set, new_test_set = {}, {}
            self.n_train, self.n_test=0, 0
            self.exist_users = []
            for k in sampled_train_set:
                new_k = user_index[k]
                items = sampled_train_set[k]
                self.exist_users.append(new_k)
                item_list = []
                for i in items:
                    new_item = item_index[i]
                    item_list.append(new_item)
                self.n_train += len(item_list)
                new_train_set[new_k] = item_list

            for k in sampled_test_set:
                new_k = user_index[k]
                items = sampled_test_set[k]
                item_list = []
                
                for i in items:
                    
                    new_item = item_index[i]
                    item_list.append(new_item)
                self.n_test += len(item_list)
                new_test_set[new_k] = item_list
            self.train_items = new_train_set
            self.test_set = new_test_set

            # Make adjacency matrix again for sampled cluster
            self.n_users, self.n_items = R.shape[0], R.shape[1]
            self.R = R
            
            adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            self.R = self.R.tolil()
            
            adj_mat[:self.n_users, self.n_users:] = self.R
            adj_mat[self.n_users:, :self.n_users] = self.R.T
            
            ### add epsilon // Zeros are not a reason why error occurs.. 
#             e_mat_u = sp.dok_matrix((self.n_users, self.n_users), dtype = np.float32)
#             e_mat_i = sp.dok_matrix((self.n_items, self.n_items), dtype = np.float32)
#             e_mat_u[:,:] = 1e-8
#             e_mat_i[:,:] = 1e-8
#             adj_mat[:self.n_users,:self.n_users] = e_mat_u
#             adj_mat[self.n_users:,self.n_users:] = e_mat_i
            
            adj_mat = adj_mat.todok()

            def normalized_adj_single(adj):
                rowsum = np.array(adj.sum(1))

                d_inv = np.power(rowsum, -1).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat_inv = sp.diags(d_inv)

                norm_adj = d_mat_inv.dot(adj)
                # norm_adj = adj.dot(d_mat_inv)
                print('generate single-normalized adjacency matrix.')
                return norm_adj.tocoo()

            def get_D_inv(adj):
                rowsum = np.array(adj.sum(1))

                d_inv = np.power(rowsum, -1).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat_inv = sp.diags(d_inv)
                return d_mat_inv

            def check_adj_if_equal(adj):
                dense_A = np.array(adj.todense())
                degree = np.sum(dense_A, axis=1, keepdims=False)

                temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
                print('check normalized adjacency matrix whether equal to this laplacian matrix.')

            self.print_statistics()
            print("Clustering Completed")
            norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
            mean_adj_mat = normalized_adj_single(adj_mat)
            norm_adj_mat = norm_adj_mat.tocsr()
            
            adj_mat = adj_mat.tocsr()
            mean_adj_mat = mean_adj_mat.tocsr()
            
            
        else : 
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
            print('already load adj matrix', adj_mat.shape, time() - t1)
            print('Orgiginal /// n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))
            

        return adj_mat, norm_adj_mat, mean_adj_mat
            

        
    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        
        R = self.R.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def get_D_inv(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            return d_mat_inv

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr(), R.tocsr()

    def negative_pool(self):
        t1 = time()
        for u in self.train_items.keys():
            neg_items = list(set(range(self.n_items)) - set(self.train_items[u]))
            pools = [rd.choice(neg_items) for _ in range(100)]
            self.neg_pools[u] = pools
        print('refresh negative pools', time() - t1)

    def sample(self):

        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]
        
        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]
                
                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
#                 neg_id = np.random.choice(self.item_list , size=1)[0]
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)
            
        return users, pos_items, neg_items

    def sample_all_users_pos_items(self):
        self.all_train_users = []

        self.all_train_pos_items = []
        for u in self.exist_users:
            self.all_train_users += [u] * len(self.train_items[u])
            self.all_train_pos_items += self.train_items[u]

    def epoch_sample(self):
        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        neg_items = []
        for u in self.all_train_users:
            neg_items += sample_neg_items_for_u(u,1)

        perm = np.random.permutation(len(self.all_train_users))
        users = np.array(self.all_train_users)[perm]
        pos_items = np.array(self.all_train_pos_items)[perm]
        neg_items = np.array(neg_items)[perm]
        return users, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))


    def get_sparsity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open(self.path + '/sparsity.split', 'r').readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split(' ')])
            print('get sparsity split.')

        except Exception:
            split_uids, split_state = self.create_sparsity_split()
            f = open(self.path + '/sparsity.split', 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create sparsity split.')

        return split_uids, split_state



    def create_sparsity_split(self):
        all_users_to_test = list(self.test_set.keys())
        user_n_iid = dict()

        # generate a dictionary to store (key=n_iids, value=a list of uid).
        for uid in all_users_to_test:
            train_iids = self.train_items[uid]
            test_iids = self.test_set[uid]

            n_iids = len(train_iids) + len(test_iids)

            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)
        split_uids = list()

        # split the whole user set into four subset.
        temp = []
        count = 1
        fold = 4
        n_count = (self.n_train + self.n_test)
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' %(n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)



        return split_uids, split_state

