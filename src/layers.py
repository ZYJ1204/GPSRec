import dill
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self,in_features,out_features,dropout,alpha,device=torch.device('cuda:0')):
        super(GATLayer,self).__init__()
        self.in_features=in_features    # Input feature dimension
        self.out_features=out_features  # Output feature
        self.dropout=dropout            # dropout ratio
        self.alpha=alpha                # Leakyrelu negative slope coefficient
        self.device = device

        # Define trainable parameters, W and a in the paper
        self.W=nn.Parameter(torch.zeros(size=(in_features,out_features))).to(device)
        nn.init.xavier_uniform_(self.W.data, gain=1.414) # xavier initialization
        self.a=nn.Parameter(torch.zeros(size=(2*out_features,1))).to(device)
        nn.init.xavier_uniform_(self.a.data, gain=1.414) # xavier initialization

        # Define leakyrelu activation function
        self.leakyrelu=nn.LeakyReLU(self.alpha)

    def forward(self,input_h,adj):
        """
        input_h: node features
        adj: adjacency matrix
        """
        # Calculate the number of nodes
        N=input_h.size()[0]
        Wh=(torch.mm(input_h,self.W)).to(self.device)
        # Concatenate
        input_concat=torch.cat([Wh.repeat(1,N).view(N*N,-1),Wh.repeat(N,1)],dim=1).view(N,-1,2*self.out_features)
        # calc eij  [N,N,2*out_features]->[N,N,1]->[N,N]
        e=self.leakyrelu(torch.matmul(input_concat,self.a).squeeze(2))

        zero_vec=-1e12*torch.ones_like(e) # Set the edges without connection to negative infinity
        attention=torch.where(adj>0,e,zero_vec) # [N,N]
        # Represents that if the adjacency matrix element is greater than 0, eij is retained, otherwise negative infinity is retained
        # That is, if there is no connection between two nodes, the attention weight between them is negative infinity
        attention=F.softmax(attention,dim=1) # softmax shape remains unchanged[N,N], get normalized attention weight
        attention=F.dropout(attention,self.dropout,training=self.training) # dropout, prevent overfitting
        out_h=torch.matmul(attention,Wh) #[N,N]*[N,out_features]->[N,out_features]
        # Get the node representation updated by the surrounding nodes through the attention weight
        return out_h

# Multi-view GAT network
class MultiViewGAT(nn.Module):
    def  __init__(self,in_features,out_features,dropout,alpha,nheads,device=torch.device('cuda:0')):
        super(MultiViewGAT,self).__init__()
        self.dropout=dropout
        self.alpha=alpha
        self.nheads=nheads
        self.attentions=[GATLayer(in_features,out_features,dropout=self.dropout,alpha=self.alpha) for _ in range(self.nheads)]
        self.device = device
        for i,attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i),attention)
        # Define the output attention layer
        self.out_att=GATLayer(out_features,out_features,dropout=self.dropout,alpha=self.alpha)
        self.res = nn.Parameter(torch.zeros(size=(1,1))).to(device)
        self.layernorm = nn.LayerNorm(out_features).to(device)

    def forward(self,input_hs,adjs,MPNN_emb,query):
        """
        input_hs: node features of multiple views
        adjs: adjacency matrix of multiple views
        """
        head_outs=[att(input_h,adj) for att,input_h,adj in zip(self.attentions,input_hs,adjs)]
        # Calculate w1,w2,w3
        w1 = torch.matmul(self.layernorm(query), head_outs[0].t())
        w2 = torch.matmul(self.layernorm(query), head_outs[1].t())
        w3 = torch.matmul(self.layernorm(query), MPNN_emb.t())
        # Calculate the attention weight
        attention_scores = F.softmax(torch.cat([w1,w2,w3],dim = 0), dim=0)
        fused_tensor = self.res * (torch.matmul(torch.diag_embed(attention_scores[0,:]), head_outs[0]) \
                        + torch.matmul(torch.diag_embed(attention_scores[1,:]), head_outs[1]) \
                        + torch.matmul(torch.diag_embed(attention_scores[2,:]), MPNN_emb) )+ input_hs[0]
        return fused_tensor

class MolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, N_fingerprint, dim, layer_hidden, device):
        super(MolecularGraphNeuralNetwork, self).__init__()
        self.device = device
        self.embed_fingerprint = nn.Embedding(N_fingerprint, dim).to(self.device)
        self.W_fingerprint = nn.ModuleList(
            [nn.Linear(dim, dim).to(self.device) for _ in range(layer_hidden)]
        )
        self.layer_hidden = layer_hidden

    def pad(self, matrices, pad_value):
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.FloatTensor(np.zeros((M, N))).to(self.device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i : i + m, j : j + n] = matrix
            i += m
            j += n
        return pad_matrices

    def update(self, matrix, vectors, layer): # update
        hidden_vectors = torch.relu(self.W_fingerprint[layer](vectors))
        return hidden_vectors + torch.mm(matrix, hidden_vectors)

    def sum(self, vectors, axis): # sum
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)

    def mean(self, vectors, axis): # mean
        mean_vectors = [torch.mean(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(mean_vectors)

    def forward(self, inputs):

        """Cat or pad each input data for batch processing."""
        fingerprints, adjacencies, molecular_sizes = inputs # molecular fingerprint, adjacency matrix, molecular size
        fingerprints = torch.cat(fingerprints)
        adjacencies = self.pad(adjacencies, 0)

        fingerprint_vectors = self.embed_fingerprint(fingerprints) # Embedding
        for l in range(self.layer_hidden):
            hs = self.update(adjacencies, fingerprint_vectors, l)
            fingerprint_vectors = hs

        molecular_vectors = self.sum(fingerprint_vectors, molecular_sizes)
        return molecular_vectors

if __name__ == '__main__':
    pass