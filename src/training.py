import os
import torch
import numpy as np
from torch.optim import Adam
from util import llprint, multi_label_metric, ddi_rate_score, get_n_params, Regularization
import time
from collections import defaultdict
from torch.nn.functional import binary_cross_entropy_with_logits, sigmoid
from torch.nn.functional import multilabel_margin_loss
from graph import draw_curve
import math
import dill


def eval_one_epoch(model, data_eval, voc_size, drug_data):
    model = model.eval()
    smm_record, ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(6)]
    med_cnt, visit_cnt = 0, 0
    for step, input_seq in enumerate(data_eval):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        for adm_idx, adm in enumerate(input_seq):
            output, _ = model(
                patient_data=input_seq[:adm_idx + 1],
                **drug_data
            )
            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            output = torch.sigmoid(output).detach().cpu().numpy()[0]
            y_pred_prob.append(output)

            y_pred_tmp = output.copy()
            y_pred_tmp[y_pred_tmp >= 0.5] = 1
            y_pred_tmp[y_pred_tmp < 0.5] = 0
            y_pred.append(y_pred_tmp)

            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)
        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
            np.array(y_gt), np.array(y_pred), np.array(y_pred_prob)
        )
        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint('\rtest step: {} / {}'.format(step + 1, len(data_eval)))

    ddi_rate = ddi_rate_score(smm_record, path='../data/output/mimic-iii/ddi_A_final.pkl')
    output_str = '\nDDI Rate: {:.4f}, Jaccard: {:.4f}, PRAUC: {:.4f}, ' +\
        'AVG_PRC: {:.4f}, AVG_RECALL: {:.4f}, AVG_F1: {:.4f}, AVG_MED: {:.4f}\n'
    llprint(output_str.format(
        ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p),
        np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt
    ))
    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), \
        np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt


def pre_eval(model, data_eval, voc_size, epoch, device):
    model.eval()

    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    loss_bce, loss_multi, loss = [[] for _ in range(3)]

    med_cnt, visit_cnt = 0, 0

    for step, input in enumerate(data_eval):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        for adm_idx, adm in enumerate(input):
            target_output, _ = model(input[: adm_idx + 1])
            loss_bce_target = np.zeros((1, voc_size[2]))
            loss_bce_target[:, adm[2]] = 1

            loss_multi_target = np.full((1, voc_size[2]), -1)
            for idx, item in enumerate(adm[2]):
                loss_multi_target[0][idx] = item

            with torch.no_grad():
                loss_bce1 = binary_cross_entropy_with_logits(
                    target_output, torch.FloatTensor(loss_bce_target).to(device)
                ).cpu()
                loss_multi1 = multilabel_margin_loss(
                    sigmoid(target_output), torch.LongTensor(loss_multi_target).to(device)
                ).cpu()
                loss1 = 0.95 * loss_bce1.item() + 0.05 * loss_multi1.item() # loss

            loss_bce.append(loss_bce1)
            loss_multi.append(loss_multi1)
            loss.append(loss1)

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            # prediction prod
            target_output = sigmoid(target_output).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output)

            # prediction med set
            y_pred_tmp = target_output.copy() # ·
            y_pred_tmp[y_pred_tmp >= 0.5] = 1
            y_pred_tmp[y_pred_tmp < 0.5] = 0
            y_pred.append(y_pred_tmp)

            # prediction label
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)

        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
            np.array(y_gt), np.array(y_pred), np.array(y_pred_prob)
        )

        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint("\rtest step: {} / {}".format(step, len(data_eval)))

    # ddi rate
    ddi_rate = ddi_rate_score(smm_record, path="../data/output/mimic-iii/ddi_A_final.pkl")

    llprint(
        "\nDDI Rate: {:.4}, Jaccard: {:.4},  PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4},"
        "AVG_Loss: {:.4}, AVG_MED: {:.4}\n".format(
            ddi_rate,
            np.mean(ja),
            np.mean(prauc),
            np.mean(avg_p),
            np.mean(avg_r),
            np.mean(avg_f1),
            np.mean(loss),
            med_cnt / visit_cnt,
        )
    )

    return (
        ddi_rate,
        np.mean(ja),
        np.mean(prauc),
        np.mean(avg_p),
        np.mean(avg_r),
        np.mean(avg_f1),
        np.mean(loss),
        med_cnt / visit_cnt,
    )

def Test(model, model_path, device, data_test, voc_size, drug_data):
    with open(model_path, 'rb') as Fin: # resume_path
        model.load_state_dict(torch.load(Fin, map_location=device),strict=False) # load model
    model = model.to(device).eval() # model to device
    print('--------------------Begin Testing--------------------')
    ddi_list, ja_list, prauc_list, f1_list, med_list = [], [], [], [], []
    tic, result, sample_size = time.time(), [], round(len(data_test) * 0.8)
    np.random.seed(0)
    for _ in range(2):
        # test_sample = np.random.choice(data_test, size=sample_size, replace=True)
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = \
            eval_one_epoch(model, data_test, voc_size, drug_data)
        result.append([ddi_rate, ja, avg_f1, prauc, avg_med])
    result = np.array(result)
    mean, std = result.mean(axis=0), result.std(axis=0)
    metric_list = ['ddi_rate', 'ja', 'avg_f1', 'prauc', 'med']
    outstring = ''.join([
        "{}:\t{:.4f} $\\pm$ {:.4f} & \n".format(metric_list[idx], m, s)
         for idx, (m, s) in enumerate(zip(mean, std))
    ])
    print(outstring)
    print('average test time: {}'.format((time.time() - tic) / 10))
    print('parameters', get_n_params(model))


def pre_training(pretrained_model,data_train,data_eval,voc_size,lr,pretrain,Test,resume_path_pretrained,target_ddi,pre_coef,device):

    if not pretrain or Test:
        pretrained_model.load_state_dict(torch.load(resume_path_pretrained, map_location=device),strict=False)
        return pretrained_model

    else:
        regular = Regularization(pretrained_model, 0.005, p=0)  # regularization
        optimizer = Adam(list(pretrained_model.parameters()), lr=lr)

        # start iterations
        best_epoch, best_ja = 0, 0
        best_model = None

        EPOCH = 20
        for epoch in range(EPOCH):
            print("\nepoch {} --------------------------".format(epoch))

            pretrained_model.train()
            for step, int_seq in enumerate(data_train):
                loss = 0
                for adm_idx, adm in enumerate(int_seq):
                    loss_bce_target = torch.zeros((1, voc_size[2])).to(device)
                    loss_bce_target[:, adm[2]] = 1

                    loss_multi_target = -torch.ones((1, voc_size[2])).long()
                    for idx, item in enumerate(adm[2]):
                        loss_multi_target[0][idx] = item
                    loss_multi_target = loss_multi_target.to(device)

                    result, loss_ddi = pretrained_model(int_seq[:adm_idx + 1])

                    loss_bce = binary_cross_entropy_with_logits(
                        result,loss_bce_target
                    )
                    loss_multi = multilabel_margin_loss(
                        sigmoid(result), loss_multi_target
                    )

                    result = sigmoid(result).detach().cpu().numpy()[0]
                    result[result >= 0.5] = 1
                    result[result < 0.5] = 0
                    y_label = np.where(result == 1)[0]
                    current_ddi_rate = ddi_rate_score(
                        [[y_label]], path="../data/output/mimic-iii/ddi_A_final.pkl"
                    )

                    if current_ddi_rate <= (target_ddi-0.02):  # 如果当前的DDI小于目标DDI
                        loss = 0.95 * loss_bce + 0.05 * loss_multi  # 以二分类交叉熵损失为主
                    else:
                        beta = pre_coef * (1.02 + abs(current_ddi_rate - target_ddi))  # 计算beta
                        beta = max(math.exp(-beta), 0)
                        loss = beta * (0.95 * loss_bce + 0.05 * loss_multi) + (1 - beta) * loss_ddi

                    loss += regular(pretrained_model)  # 正则化

                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

                llprint("\rtraining step: {} / {}".format(step, len(data_train)))
            ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_loss, avg_med = pre_eval(
                pretrained_model, data_eval, voc_size, epoch, device
            )
            print('\n 测试集结果：')
            if epoch != 0:
                if best_ja < ja:
                    best_epoch = epoch
                    best_ja = ja
                    best_model = pretrained_model
                print(
                    "best_epoch: {}, best_ja: {:.4}".format(best_epoch, best_ja))
        #
        torch.save(best_model.state_dict(), resume_path_pretrained)
        return best_model


def Train(
    model, device, data_train, data_eval, voc_size, drug_data,
    optimizer, log_dir, coef, target_ddi, EPOCH=50         # coef: beta的系数
):
    history = defaultdict(list)
    best = {"epoch": 0, "ja": 0.0, "ddi": 0, "prauc": 0, "f1": 0, "med": 0, 'model': None}
    total_train_time, ddi_losses, ddi_values = 0, [], []
    regular = Regularization(model, 0.005, p=0)  # 正则化模型
    for epoch in range(EPOCH):
        print(f'----------------Epoch {epoch + 1}------------------')
        model = model.train()
        tic, ddi_losses_epoch = time.time(), []
        for step, input_seq in enumerate(data_train):
            loss = 0
            for adm_idx, adm in enumerate(input_seq):
                bce_target = torch.zeros((1, voc_size[2])).to(device)
                bce_target[:, adm[2]] = 1

                multi_target = -torch.ones((1, voc_size[2])).long()
                for idx, item in enumerate(adm[2]):
                    multi_target[0][idx] = item
                multi_target = multi_target.to(device)
                # 计算模型输出和DDI损失
                result, loss_ddi = model(
                    patient_data=input_seq[:adm_idx + 1],
                    **drug_data
                )

                sigmoid_res = torch.sigmoid(result)

                loss_bce = binary_cross_entropy_with_logits(result, bce_target) # 二分类交叉熵损失
                loss_multi = multilabel_margin_loss(sigmoid_res, multi_target)  # 多标签损失

                result = sigmoid_res.detach().cpu().numpy()[0]
                result[result >= 0.5] = 1
                result[result < 0.5] = 0
                y_label = np.where(result == 1)[0] # 预测的标签
                current_ddi_rate = ddi_rate_score( # 计算当前的DDI
                    [[y_label]], path='../data/output/mimic-iii/ddi_A_final.pkl'
                )

                if current_ddi_rate <= (target_ddi-0.04): # $\theta_2$=0.01
                    loss = 0.95 * loss_bce + 0.05 * loss_multi # 以二分类交叉熵损失为主
                else:
                    beta = coef * (1.02 + abs(current_ddi_rate - target_ddi)) # 计算beta
                    beta = max(math.exp(-beta), 0)
                    loss = beta * (0.95 * loss_bce + 0.05 * loss_multi) + (1 - beta) * loss_ddi

                loss += regular(model)  # 跟上面配套
                ddi_losses_epoch.append(loss_ddi.detach().cpu().item()) # 记录DDI损失
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            llprint('\rtraining step: {} / {}'.format(step, len(data_train)))
        ddi_losses.append(sum(ddi_losses_epoch) / len(ddi_losses_epoch))
        # print(f'\nddi_loss : {ddi_losses[-1]}\n')
        train_time, tic = time.time() - tic, time.time()
        total_train_time += train_time
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = \
            eval_one_epoch(model, data_eval, voc_size, drug_data)
        print(f'training time: {train_time}, testing time: {time.time() - tic}')
        ddi_values.append(ddi_rate)
        history['ja'].append(ja)
        history['ddi_rate'].append(ddi_rate)
        history['avg_p'].append(avg_p)
        history['avg_r'].append(avg_r)
        history['avg_f1'].append(avg_f1)
        history['prauc'].append(prauc)
        history['med'].append(avg_med)

        if epoch != 0:
            if best['ja'] < ja:
                best['epoch'] = epoch
                best['ja'] = ja
                best['model'] = model
                best['ddi'] = ddi_rate
                best['prauc'] = prauc
                best['f1'] = avg_f1
                best['med'] = avg_med
            print( "best_epoch: {}, best_ja: {:.4}, best_ddi: {:.4}".format(best['epoch']+1, best['ja'], best['ddi']))

        torch.save(model.state_dict(), os.path.join(log_dir, 'Epoch_{}_JA_{:.4f}_DDI_{:.4f}.model'.format(
                                                                            epoch, ja, ddi_rate)
        ))
        with open(os.path.join(log_dir, 'ddi_losses.txt'), 'w') as Fout:
            for dloss, dvalue in zip(ddi_losses, ddi_values):
                Fout.write(f'{dloss}\t{dvalue}\n')
        with open(os.path.join(log_dir, 'history.pkl'), 'wb') as Fout:
            dill.dump(history, Fout)
    model_name = 'Epoch_{}_JA_{:.4f}_DDI_{:.4f}_BEST.model'.format(
        best['epoch']+1, best['ja'], best['ddi']
    )
    torch.save(best['model'].state_dict(), os.path.join(log_dir, model_name))
    print('avg training time/epoch: {:.4f}'.format(total_train_time / EPOCH))
    draw_curve(log_dir, best['epoch'])

if __name__ == '__main__':
    pass