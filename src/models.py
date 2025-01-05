import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from layers import GATLayer, MultiViewGAT, MolecularGraphNeuralNetwork
import math

from src.util import cos_sim, pearson_sim, calculate_jaccard_similarity


class PretrainModel(nn.Module):
    def __init__(
            self,
            vocab_size,
            ddi_adj,
            emb_dim=64,
            device=torch.device("cpu"),
    ):
        super(PretrainModel, self).__init__()

        self.device = device
        self.emb_dim = emb_dim
        # pre-embedding
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim) for i in range(3)]
        )
        self.dropout = nn.Dropout(p=0.5)
        self.query = nn.Sequential(nn.ReLU(), nn.Linear(3 * emb_dim, vocab_size[2]))
        self.tensor_ddi_adj = ddi_adj.to(device)
        self.init_weights()

    def forward(self, patient):

        def sum_embedding(embedding):
            return embedding.sum(dim=1).unsqueeze(dim=0)  # (1,1,dim)

        adm = patient[-1]
        i1 = sum_embedding(
            self.dropout(
                self.embeddings[0](
                    torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device)
                )
            )
        )  # (1,1,dim)
        i2 = sum_embedding(
            self.dropout(
                self.embeddings[1](
                    torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device)
                )
            )
        )
        if len(patient) <= 1:
            i3 = torch.zeros((1, 1, self.emb_dim)).to(self.device)
        else:
            adm = patient[-2]
            i3 = sum_embedding(
                self.dropout(
                    self.embeddings[2](
                        torch.LongTensor(adm[2]).unsqueeze(dim=0).to(self.device)
                    )
                )
            )

        patient_representations = torch.cat([i1, i2, i3], dim=-1).squeeze(
            dim=0
        )  # (seq, dim*2)
        result = self.query(patient_representations)[-1:, :]  # (seq, dim)

        neg_pred_prob = F.sigmoid(result)
        neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)

        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

        return result, batch_neg

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)

class GPSRecModel(nn.Module):
    def __init__(
        self,emb_dim, voc_size, drug_num, MPNNSet, N_fingerprints, average_projection,
        device=torch.device('cpu'), dropout=0.5,pretrained_embedding=None,*args, **kwargs
    ):
        super(GPSRecModel, self).__init__(*args, **kwargs)
        self.device = device
        self.voc_size = voc_size
        self.embeddings = nn.ModuleList([       # diagnosis, procedure, medication embeddings
            nn.Embedding(voc_size[0], emb_dim),
            nn.Embedding(voc_size[1], emb_dim),
            nn.Embedding(voc_size[2], emb_dim)
        ])
        # molecule embedding
        self.MPNN_molecule_Set = list(zip(*MPNNSet))
        self.MPNN_emb = MolecularGraphNeuralNetwork(N_fingerprints, emb_dim, layer_hidden=2, device=device).forward(self.MPNN_molecule_Set)
        self.MPNN_emb = torch.mm(average_projection.to(device=self.device), self.MPNN_emb.to(device=self.device))
        self.MPNN_emb.to(device=self.device)

        self.seq_encoders = nn.ModuleList([     # GRUs for diagnosis and procedure
            nn.GRU(emb_dim, emb_dim*2, batch_first=True),
            nn.GRU(emb_dim, emb_dim*2, batch_first=True)
        ])
        if dropout > 0 and dropout < 1: # dropout
            self.rnn_dropout = nn.Dropout(p=dropout)
        else:
            self.rnn_dropout = nn.Sequential()

        self.query = nn.Sequential( # query
            nn.ReLU(),
            nn.Linear(emb_dim * 4, emb_dim)
        )
        self.current_visit = nn.Linear(emb_dim, drug_num) # o_cur
        self.layernorm = nn.LayerNorm(emb_dim) # LayerNorm

        # init multi-view GAT
        self.multiviewgat = MultiViewGAT(emb_dim,emb_dim,0.2,0.2,2)

        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 3, emb_dim * 6),
            nn.ReLU(),
            nn.Linear(emb_dim * 6, voc_size[2])  # output layer
        )
        self.al1 = nn.Parameter(torch.tensor(0.5)).to(self.device)
        self.al2 = nn.Parameter(torch.tensor(0.25)).to(self.device)
        self.al3 = nn.Parameter(torch.tensor(0.25)).to(self.device)

        self.init_weights(pretrained_embedding)

    def init_weights(self, pretrained_embedding=None):
        """Initialize weights."""
        initrange = 0.1
        if pretrained_embedding is not None:
            for i, item in enumerate(self.embeddings):
                item.weight.data = pretrained_embedding[i].weight.data
        else:
            for item in self.embeddings:
                item.weight.data.uniform_(-initrange, initrange)

        self.current_visit.weight.data.uniform_(-initrange, initrange)

    def forward(
        self,tensor_ddi_adj,ehr_adj,patient_data,  # DDI adjacency matrix, EHR adjacency matrix, patient data
    ):
        seq1, seq2 = [], []
        for adm in patient_data:
            Idx1 = torch.LongTensor([adm[0]]).to(self.device)  # diagnosis
            Idx2 = torch.LongTensor([adm[1]]).to(self.device)  # procedure
            repr1 = self.rnn_dropout(self.embeddings[0](Idx1)) # diagnosis embeddings
            repr2 = self.rnn_dropout(self.embeddings[1](Idx2)) # procedure embeddings
            seq1.append(torch.sum(repr1, keepdim=True, dim=1)) # diagnosis embeddings sum
            seq2.append(torch.sum(repr2, keepdim=True, dim=1)) # procedure embeddings sum
        seq1 = torch.cat(seq1, dim=1) # concatenate diagnosis embeddings
        seq2 = torch.cat(seq2, dim=1)
        output1, hidden1 = self.seq_encoders[0](seq1) #  [1, seq_len, emb_dim]  o:(1, seq, dim*2) h:(1,1,dim*2)
        output2, hidden2 = self.seq_encoders[1](seq2) #  [1, seq_len, emb_dim]
        patient_repr = torch.cat([output1, output2], dim=-1).squeeze(dim=0)
        queries = self.query(patient_repr) # [emb_dim]

        # current visit
        query = queries[-1:]  # (1,dim)

        # multi-view GAT
        drug_embeddings = self.embeddings[2].weight.data # (131,64)
        drug_gat_embeddings = self.multiviewgat([drug_embeddings, drug_embeddings], [ehr_adj, tensor_ddi_adj],self.MPNN_emb,self.layernorm(query)) # (dim,64)

        drug_patient_weight = torch.sigmoid(self.current_visit(query))
        fact1 = torch.mm(drug_patient_weight, drug_gat_embeddings)  # (1, dim) o_cur

        if len(patient_data) > 1:
            history_visit = queries[:(queries.size(0) - 1)]  # (seq-1, dim)
            history_drug = np.zeros((len(patient_data) - 1, self.voc_size[2])) # (seq-1, size)
            for idx, adm in enumerate(patient_data):
                if idx == len(patient_data) - 1:
                    break
                history_drug[idx, adm[2]] = 1
            history_drug = torch.FloatTensor(history_drug).to(self.device)  # (seq-1, size)

        # only one visit
        if len(patient_data) == 1:
            fact2 = fact1

        if len(patient_data) >= 2:
            # s_q
            visit_weights = []
            for i in range(history_visit.size(0)):
                visit_weights.append(pearson_sim(query, history_visit[i]).view(1, 1))
            visit_weights = torch.cat(visit_weights, dim=1).to(self.device)  # (1, seq-1)
            denominator = sum(range(1, history_visit.size(0) + 1))
            time_sequence = [n / denominator for n in range(1, history_visit.size(0) + 1)] # time factor
            time_sequence = torch.FloatTensor(time_sequence).to(self.device).view(1, -1)
            visit_weights = visit_weights * time_sequence

            diag_seq = [adm[0] for adm in patient_data] # diagnosis set
            pro_seq = [adm[1] for adm in patient_data] # procedure set
            # find the Jaccard similarity between the last visit and the previous visits
            diag_weights = torch.tensor(calculate_jaccard_similarity(diag_seq)).to(self.device)
            pro_weights = torch.tensor(calculate_jaccard_similarity(pro_seq)).to(self.device)
            # fusion
            visit_weights = visit_weights * self.al1 + diag_weights * self.al2 + pro_weights * self.al3

            key_weights2 = torch.mm(visit_weights, history_drug)  # (1, size[2])
            fact2 = torch.mm(key_weights2, drug_gat_embeddings)  # (1, dim)

        score = self.output(torch.cat([self.layernorm(query),fact1,fact2], dim=-1))  # (1, 3dim) -> (1, size[2])
        neg_pred_prob = torch.sigmoid(score) # (1, size[2])
        neg_pred_prob = torch.matmul(neg_pred_prob.t(), neg_pred_prob)
        batch_neg = 0.0005 * neg_pred_prob.mul(tensor_ddi_adj).sum()
        return score, batch_neg