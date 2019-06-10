# coding:utf-8

import os
from os import path
import time
import pickle as pkl
# import matplotlib.pylab as plt
from utils.data_loading import *

from config import get_arguments 

import pickle

from sklearn.metrics import roc_curve, roc_auc_score

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import torch.optim as optim

from models.mado_nets import * 

from utils.compute_corr_matrix import *

from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings("ignore")

import torch
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(1234)


config = get_arguments()

config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(config)



def train(dataset):

    train_x, train_y, train_adj, test_x, test_y, test_adj, index =  dataset

    seq_len, fea_dim = np.shape(train_x)[1], np.shape(train_x)[2]

    train_x = torch.FloatTensor(train_x).to(config.device)
    train_y = torch.LongTensor(train_y).to(config.device)
    train_adj = torch.FloatTensor(train_adj).to(config.device)

    test_x = torch.FloatTensor(test_x).to(config.device)
    test_y = torch.LongTensor(test_y).to(config.device)
    index = torch.LongTensor(index).to(config.device)
    test_adj = torch.FloatTensor(test_adj).to(config.device)

    train_loader = DataLoader(TensorDataset(train_x, train_y, train_adj), batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_x, test_y, test_adj, index), batch_size=256)
    print(test_x.shape, test_y.shape, test_adj.shape)

    mynet = MADO_Net(config=config, seq_len=seq_len, fea_dim=fea_dim, mode=config.mode)
    mynet = mynet.to(config.device)

    # Set optimizer (Adam optimizer for now)
    optimizer = optim.Adam(mynet.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,next_prediction T_max=5,eta_min=4e-08)
    # scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma = 0.9)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[20,40],gamma = 0.9)
    
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=4, verbose=True)
    # factor(float)- 学习率调整倍数(等同于其它方法的 gamma)，即学习率更新为 lr = lr * factor
    # patience(int)- 忍受该指标多少个 step 不变化，当忍无可忍时，调整学习率。
    print(config.load_model)
    if config.load_model:
        
        # mynet = torch.load("log/" + config.dataset_name + "_model")
        model_dict = torch.load("log/" + config.dataset_name+"_"+config.mode+ "_model.tar")
        mynet.c = model_dict['c']
        mynet.load_state_dict(model_dict['net_dict'])

        # '''

        idx_label_score = []
        mynet.eval()
        with torch.no_grad():
            count = 0
            for test_data in test_loader:
                # print("processing {} test batch ...".format(count))
                inputs, labels, adj, index = test_data
                inputs = inputs.to(config.device)
                adj = adj.to(config.device)
                outputs = mynet(inputs, adj)
                seq_len = len(outputs)

                if mynet.add_learned_r:
                    dim_expand_r = outputs[0].shape[0]
                    dist_0 = torch.zeros(dim_expand_r, 1).to(config.device)

                for i in range(seq_len):
                    """
                    if mynet.add_learned_r:
                        # print("haha", np.shape(outputs[i])) # [64, 32] batchsize 64
                        dist = torch.sum((outputs[i] - mynet.c) ** 2, dim=1)
                        dist = torch.unsqueeze(dist, dim=1)
                        dist = dist - mynet.adjusted_r

                        dist = torch.cat([dist_0, dist], dim=1)
                        dist = torch.max(dist, 1)[0]
                    else:
                        dist = torch.sum((outputs[i] - mynet.c) ** 2, dim=1)
                    """
                    dist = torch.sum((outputs[i] - mynet.c) ** 2, dim=1)
                    scores = dist

                    idx_label_score += list(zip(scores.cpu().data.numpy().tolist(),
                                                labels[:, i].cpu().data.numpy().tolist(),
                                                scores.cpu().data.numpy().tolist(),
                                                index[:, i].cpu().data.numpy().tolist()))
                count += 1


        _, labels, scores, index = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        index = np.array(index)

        labels, scores = merge_repeat_records(labels, scores, index)

        # with open("log/" + config.dataset_name + "_score.pkl", 'wb') as f:
        #     pickle.dump([scores], f)
        # print(np.array(scores).shape)

        # raise Exception("testing here!")

        auc = roc_auc_score(labels, scores)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, scores)

        # with open("/home/lee/PycharmProjects/codes 1.4/roc_"+ config.dataset_name +"_taonet", 'wb') as f:
        #     pickle.dump([false_positive_rate, true_positive_rate, auc], f)

        with open("log/" + config.dataset_name + "_" + config.mode + "_auc.pkl", 'wb') as f:
            pickle.dump([false_positive_rate, true_positive_rate, auc], f)

        print("auc=", auc)
        quit()

        from sklearn.metrics import recall_score
        from sklearn.metrics import f1_score
        from sklearn.metrics import precision_score
        from sklearn.metrics import accuracy_score

        best_result = [0, 0, 0]
        sort_index = np.argsort(scores)

        for i in range(0, 100):
            tao = scores[sort_index[int((0.01 * i) * scores.shape[0])]]
            # tao = np.min(scores) + (np.max(scores) - np.min(scores)) * i/80
            scores_tao = [1 if x > tao else 0 for x in scores]
            Pre = precision_score(labels, scores_tao)
            Rec = recall_score(labels, scores_tao)
            F1 = f1_score(labels, scores_tao)
            if F1 > best_result[2]:
                best_result = [Pre, Rec, F1]

                # print('Test set Pre: {:.4}%; Rec: {:.4}%; F1: {:.4}%; tao={:.3}'.format(100. * Pre, 100. * Rec, 100. * F1, tao))

        print('The best result set Pre: {:.4}%; Rec: {:.4}%; F1: {:.4}%'.format(100. * best_result[0],
                                                                                100. * best_result[1],
                                                                                100. * best_result[2]))
        quit()
        # '''

    if mynet.c is None:
        mynet.c = init_center_c(config, mynet, DataLoader(TensorDataset(train_x, train_y, train_adj), batch_size=config.batch_size))

    current_loss = 0

    best_F1, best_pre, best_rec = 0, 0, 0
    # with open("log/" + config.dataset_name+"_"+config.mode+ "_f1.pkl", 'rb') as f:
    #     best_F1 = pickle.load(f)


    best_auc = 0.0
    best_model = None
    best_model_c = None
    for epoch in range(0, config.num_epoch+1):

        mynet.train()
        n_batches = 0
        loss_epoch = 0.0
        epoch_start_time = time.time()
        for data in train_loader:

            optimizer.zero_grad()

            train_x, _, train_adj = data
            train_x.to(config.device)
            train_adj.to(config.device)

            outputs = mynet(train_x, train_adj)
            loss = mynet.computing_oneclass_loss(config, outputs)

            loss.backward()
            optimizer.step()

            loss_epoch += loss.item()
            n_batches += 1

        epoch_train_time = time.time() - epoch_start_time
       
        current_loss = loss_epoch / n_batches
        
        print('Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'.format(epoch+1 , config.num_epoch, epoch_train_time, current_loss))


        # if epoch % 1==0 and epoch>0 and epoch<51:
        #     adjust_learning_rate(optimizer, rate=0.75)

        if epoch % 5 == 0 and epoch > 0 and epoch < 101 and config.dataset_name == "swat":
            adjust_learning_rate(optimizer, rate=0.95)

        if epoch % 10==0 and epoch>0 and epoch<51 and config.dataset_name == "wadi":
            adjust_learning_rate(optimizer, rate=0.75)
        
        # scheduler.step()

        if epoch % 1 == 0:
            idx_label_score = []
            mynet.eval()
            with torch.no_grad():
                count = 0
                for test_data in test_loader:
                    # print("processing {} test batch ...".format(count))
                    inputs, labels, adj, index = test_data
                    inputs = inputs.to(config.device)
                    adj = adj.to(config.device)
                    outputs = mynet(inputs, adj)
                    seq_len = len(outputs)

                    if mynet.add_learned_r:
                        dim_expand_r = outputs[0].shape[0]
                        dist_0 = torch.zeros(dim_expand_r, 1).to(config.device)

                    for i in range(seq_len):
                        """
                        if mynet.add_learned_r:
                            # print("haha", np.shape(outputs[i])) # [64, 32] batchsize 64
                            dist = torch.sum((outputs[i] - mynet.c) ** 2, dim=1)
                            dist = torch.unsqueeze(dist, dim=1)
                            dist = dist - mynet.adjusted_r

                            dist = torch.cat([dist_0, dist], dim=1)
                            dist = torch.max(dist, 1)[0]
                        else:
                            dist = torch.sum((outputs[i] - mynet.c) ** 2, dim=1)
                        """
                        dist = torch.sum((outputs[i] - mynet.c) ** 2, dim=1)
                        scores = dist

                        idx_label_score += list(zip(scores.cpu().data.numpy().tolist(),
                                                    labels[:, i].cpu().data.numpy().tolist(),
                                                    scores.cpu().data.numpy().tolist(),
                                                    index[:, i].cpu().data.numpy().tolist()))

                    count += 1

            _, labels, scores, index = zip(*idx_label_score)
            labels = np.array(labels)
            scores = np.array(scores)
            index = np.array(index)

            auc = roc_auc_score(labels, scores)

            labels, scores = merge_repeat_records(labels, scores, index)
            
            # raise Exception("testing here!") 

            from sklearn.metrics import recall_score
            from sklearn.metrics import f1_score
            from sklearn.metrics import precision_score

            best_result = [0, 0, 0]
            sort_index = np.argsort(scores)

            for i in range(0, 100):
                tao = scores[sort_index[int((0.01 * i) * scores.shape[0])]]
                # tao = np.min(scores) + (np.max(scores) - np.min(scores)) * i/80
                scores_tao = [1 if x > tao else 0 for x in scores]
                Pre = precision_score(labels, scores_tao)
                Rec = recall_score(labels, scores_tao)
                F1 = f1_score(labels, scores_tao)
                if F1 > best_result[2]:
                    best_result = [Pre, Rec, F1]

                # print('Test set Pre: {:.4}%; Rec: {:.4}%; F1: {:.4}%; tao={:.3}'.format(100. * Pre, 100. * Rec, 100. * F1, tao))

            print('The best result set Pre: {:.4}%; Rec: {:.4}%; F1: {:.4}%'.format(100. * best_result[0], 100. * best_result[1], 100. * best_result[2]))
            print("best_auc =", auc)

            if best_result[2] > best_F1:
                best_F1, best_pre, best_rec = best_result[2], best_result[0], best_result[1]

                # best_model = mynet
                best_model = mynet.state_dict()
                best_model_c = mynet.c

                if config.save_model:
                    # torch.save(best_model, "log/"+config.dataset_name+"_model")
                    net_dict = best_model
                    torch.save({'c': best_model_c, 'net_dict': net_dict}, "log/" + config.dataset_name+"_"+config.mode+ "_model.tar")
                    with open("log/" + config.dataset_name+"_"+config.mode+ "_f1.pkl", 'wb') as f:
                        pickle.dump(best_F1, f)

            if auc > best_auc:
                best_auc = auc

                false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, scores)

                with open("log/" + config.dataset_name+"_"+config.mode+ "_auc.pkl", 'wb') as f:
                    pickle.dump([false_positive_rate, true_positive_rate, auc], f)



    print('The best result set Pre: {:.4}%; Rec: {:.4}%; F1: {:.4}%'.format(100. * best_pre, 100. * best_rec, 100. * best_F1))
    print('The best result set best_auc:{:.4}'.format(best_auc))


"""
def add_l2_loss(model, loss):
    weight_l2 = torch.tensor(0.01)
    l2_reg = torch.tensor(0.)
    for param in model.parameters():
        l2_reg += torch.norm(param)
    loss += weight_l2 * l2_reg

    return loss
"""

def adjust_learning_rate(optimizer, rate=0.8):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr']*rate


def merge_repeat_records(labels, scores, index):

    n, m = np.shape(index)

    res = np.zeros((n, 3))

    for i in range(n):
        res[i,0] = index[i]
        res[i,1] = labels[i]
        res[i,2] = scores[i]

    res = pd.DataFrame(res)

    # print(list(res.columns.values))
    # print(res.shape[0])

    res = res.groupby(0).agg({0:'mean', 1:'mean', 2:'mean'})

    # print(res.shape[0])

    res = np.array(res)

    labels = res[:,1]
    scores = res[:,2]

    return labels, scores


def init_center_c(config, net, train_loader, state=True):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    n_samples = 0
    eps=0.1

    c = torch.zeros(net.rep_dim).to(config.device)

    # net.eval()
    with torch.no_grad():
        count = 0
        for data in train_loader:
            # get the inputs of the batch
            if state is True:
                print("initialize center vectors: using {}-th samples...".format(count))
            inputs, _, adj = data
            inputs = inputs.to(config.device)
            adj = adj.to(config.device)
            outputs = net(inputs, adj)
            seq_len = len(outputs)
            for i in range(seq_len):
                c += torch.sum(outputs[i], dim=0)
            n_samples += outputs[i].shape[0] * seq_len

            count += np.shape(inputs)[0]

    c /= n_samples

    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    # c = torch.Size([32])

    return c


def main():

    if not path.isdir(config.log_dir):
        os.makedirs(config.log_dir)

    if not path.isdir(config.processed_data_dir):
        os.makedirs(config.processed_data_dir)

    dataset_name = config.dataset_name
    train_x, train_y,  = main_loading_dataset(config, dataset_name, train_or_test="train")
    test_x, test_y, index = main_loading_dataset(config, dataset_name, train_or_test="test")

    # (56226, 120, 34)
    # print(np.shape(train_x))
    
    if not os.path.exists(os.path.join(config.processed_data_dir, "{}_adj_train_x.h5".format(dataset_name))):
        
        adj_train_x = compute_corr_matrix(train_x)
        
        f = h5py.File(os.path.join(config.processed_data_dir, "{}_adj_train_x.h5".format(dataset_name)),'w')
        f["adj"] = adj_train_x
        f.close()
        print("saving file {}_adj_train_x.h5 successfully!".format(dataset_name))
    else:
        f = h5py.File(os.path.join(config.processed_data_dir, "{}_adj_train_x.h5".format(dataset_name)),'r')
        adj_train_x = f["adj"][:]
        f.close()
        print("loading file {}_adj_train_x.h5 successfully!".format(dataset_name))


    if not os.path.exists(os.path.join(config.processed_data_dir, "{}_adj_test_x.h5".format(dataset_name))):
        
        adj_test_x = compute_corr_matrix(test_x)
        
        f = h5py.File(os.path.join(config.processed_data_dir, "{}_adj_test_x.h5".format(dataset_name)),'w')
        f["adj"] = adj_test_x
        f.close()
        print("saving file {}_adj_test_x.h5 successfully!".format(dataset_name))
    else:
        f = h5py.File(os.path.join(config.processed_data_dir, "{}_adj_test_x.h5".format(dataset_name)),'r')
        adj_test_x = f["adj"][:]
        f.close()
        print("loading file {}_adj_test_x.h5 successfully!".format(dataset_name))

    dataset = [train_x, train_y, adj_train_x, test_x, test_y, adj_test_x, index]

    train(dataset)    





if __name__ == '__main__':
    main()





