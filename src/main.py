import os
import torch
import numpy as np
import dill
import argparse
from torch.optim import Adam
from models import GPSRecModel, PretrainModel
from training import Test, Train, pre_training, eval_one_epoch
from util import buildMPNN, get_n_params

torch.manual_seed(1203)
np.random.seed(2048)

def parse_args():

    parser = argparse.ArgumentParser('Experiment For GPSRec')
    parser.add_argument('--Test', default=True, help="evaluating mode")
    parser.add_argument('--pretrain', default=False, type=bool, help='pretrain mode')
    parser.add_argument('--dim', default=64, type=int, help='model dimension')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--dp', default=0.7, type=float, help='dropout ratio')
    parser.add_argument(
        '--model_name', type=str,default='GPSRec_0.01_0.05_finally',
        help="the model name for training, if it's left blank,"
        " the code will generate a new model name on it own"
    )
    parser.add_argument(
        '--resume_path', type=str,default='../saved/GPSRec_0.01_0.05_finally/Epoch_5_JA_0.5372_DDI_0.0499.model',
        help='path of well pretrain model, only for evaluating the model'
    )
    parser.add_argument(
        '--resume_path_pretrained', type=str, default='../pretrain/pretrained_model.model',
        help='path of well pretrain model, only for evaluating the model'
    )
    parser.add_argument(
        '--device', type=int, default=0,
        help='gpu id to run on, negative for cpu'
    )
    parser.add_argument(
        '--target_ddi', type=float, default=0.05,
        help='expected ddi for training'
    )
    parser.add_argument('--pre_coef', default=2, type=float, help='pretrain coef')
    parser.add_argument(
        '--coef', default=1.5, type=float,
        help='coefficient for DDI Loss Weight Annealing'
    )
    parser.add_argument(
        '--epochs', default=25,type=int,
        help='the epochs for training'
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    if not torch.cuda.is_available() or args.device < 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device}')

    data_path = '../data/output/mimic-iii/records_final.pkl'
    voc_path = '../data/output/mimic-iii/voc_final.pkl'
    ddi_adj_path = '../data/output/mimic-iii/ddi_A_final.pkl'
    ehr_adj_path = '../data/output/mimic-iii/ehr_adj_final.pkl'
    molecule_path = '../data/output/mimic-iii/atc3toSMILES.pkl'

    with open(ddi_adj_path, 'rb') as Fin:
        ddi_adj = torch.from_numpy(dill.load(Fin)).to(device) #(131,131)
    with open(ehr_adj_path, 'rb') as Fin:
        ehr_adj = torch.from_numpy(dill.load(Fin)).to(device) #(131,131)
    with open(data_path, 'rb') as Fin:
        data = dill.load(Fin)
    with open(voc_path, 'rb') as Fin:
        voc = dill.load(Fin)
    with open(molecule_path,'rb') as Fin:
        molecule = dill.load(Fin)

    diag_voc, pro_voc, med_voc = \
        voc['diag_voc'], voc['pro_voc'], voc['med_voc']

    voc_size = ( #(1958,1430,131)
        len(diag_voc.idx2word),
        len(pro_voc.idx2word),
        len(med_voc.idx2word)
    )

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point + eval_len:]
    # 筛选data_eval中长度为1的数据
    data_eval = [x for x in data_eval if len(x) == 4]

    MPNNSet, N_fingerprint, average_projection = buildMPNN(molecule, med_voc.idx2word, 2, device)

    # Pretrain Model
    pre_model = PretrainModel(voc_size, ddi_adj, emb_dim=args.dim, device=device).to(device)

    pretrained_model = pre_training(pre_model, data_train, data_eval,voc_size,args.lr, args.pretrain, args.Test,
                                    args.resume_path_pretrained, args.target_ddi, args.pre_coef, device)

    model = GPSRecModel(
        emb_dim=args.dim, drug_num=ehr_adj.shape[1], MPNNSet=MPNNSet,
        N_fingerprints=N_fingerprint, average_projection=average_projection, voc_size=voc_size,
        device=device, dropout=args.dp, pretrained_embedding= pretrained_model.embeddings
    ).to(device)
    print('parameters', get_n_params(model))
    drug_data = {
        'tensor_ddi_adj': ddi_adj,  # DDI adjacency matrix (131,131)
        'ehr_adj': ehr_adj,  # EHR adjacency matrix (131,131)
    }

    if args.Test:
        Test(model, args.resume_path, device, data_eval, voc_size, drug_data)
    else:
        if not os.path.exists(os.path.join('../saved', args.model_name)):
            os.makedirs(os.path.join('../saved', args.model_name))
        log_dir = os.path.join('../saved', args.model_name)
        optimizer = Adam(model.parameters(), lr=args.lr)
        Train(
            model, device, data_train, data_eval, voc_size, drug_data,
            optimizer, log_dir, args.coef, args.target_ddi, EPOCH=args.epochs
        )