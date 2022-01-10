#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang, partly revised by Shi Qiu (shi.qiu@anu.edu.au)
@Contact: yuewangx@mit.edu
@File: main.py
@Time: 2018/10/13 10:39 PM
"""


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNet40, ScanObjectNN
from model import PointNet, DGCNN, GBNet ,BGA
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
#from msloss import MultiSimilarityLoss, regularization_loss
import logging
import datetime 
from contrastloss import MetricLoss, categoryweightMetricLoss, entropyweightMetricLoss,WeightMetricLoss, modelnetcategoryweightMetricLoss, modelnetyweightMetricLoss



def _init_():
    
    #args = parse_args()
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))#'2021-07-27_20-47'
    if args.log_dir is None:
        experiment_dir = args.exp_name + '/'+ timestr + '/'
    else:
        experiment_dir = args.exp_name + '/'+ args.log_dir + '/'
    #experiment_dir.mkdir(exist_ok=True)
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+ args.exp_name):
        os.makedirs('checkpoints/'+ args.exp_name)
    if not os.path.exists('checkpoints/'+ experiment_dir):
        os.makedirs('checkpoints/'+ experiment_dir)
    if not os.path.exists('checkpoints/'+ experiment_dir +'/'+ 'logs'):
        log_dir = 'checkpoints/'+ experiment_dir +'/'+ 'logs'
        os.makedirs(log_dir)
    if not os.path.exists('checkpoints/'+ experiment_dir +'/'+ args.model):
        os.makedirs('checkpoints/'+ experiment_dir +'/'+ args.model)
    os.system('cp main.py checkpoints'+'/'+experiment_dir+'/'+'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + experiment_dir + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + experiment_dir + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + experiment_dir + '/' + 'data.py.backup')
   


def seg_loss( seg_pred, gt_mask):
    """ pred: BxNxC, seg_pred:[b,1,1024]
        label: BxN,gt_mask:[24,1024] """
    batch_size = gt_mask.shape[0]
    #seg_pred = F.softmax(seg_pred, dim=1)#[24,2,1024]
    seg_pred = seg_pred.view(batch_size,-1)

    maskzero = torch.zeros_like(gt_mask)
    mask1 = torch.ones_like(gt_mask)
    gt_mask = torch.where(gt_mask>-1,maskzero,gt_mask)
    gt_mask = torch.where(gt_mask == -1,mask1,gt_mask)

    
    per_instance_seg_loss = -torch.mean(gt_mask * torch.log(seg_pred)+(mask1-gt_mask)*torch.log(mask1-seg_pred), dim=1)#[24] around4000
    seg_loss = torch.mean(per_instance_seg_loss)
   

    return  seg_loss

def train(args, io):
    if args.dataset == 'modelnet40':
        train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=8,
                                  batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                                 batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    elif args.dataset == 'ScanObjectNN':
        train_loader = DataLoader(ScanObjectNN(partition='training', num_points=args.num_points), num_workers=8,
                                  batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(ScanObjectNN(partition='test', num_points=args.num_points), num_workers=8,
                                 batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    else:
        raise Exception("Dataset Not supported")
    
    #device select
    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'pointnet':
        if args.dataset == 'modelnet40':
            model = PointNet(args, output_channels=40).to(device)
        elif args.dataset == 'ScanObjectNN':
            model = PointNet(args, output_channels=15).to(device)
        else:
            raise Exception("Dataset Not supported")
    elif args.model == 'dgcnn':
        if args.dataset == 'modelnet40':
            model = DGCNN(args, output_channels=40).to(device)
        elif args.dataset == 'ScanObjectNN':
            model = DGCNN(args, output_channels=15).to(device)
        else:
            raise Exception("Dataset Not supported")
    elif args.model == 'gbnet':
        if args.dataset == 'modelnet40':
            model = GBNet(args, output_channels=40).to(device)
        elif args.dataset == 'ScanObjectNN':
            model = GBNet(args, output_channels=15).to(device)
        else:
            raise Exception("Dataset Not supported")
    elif args.model == 'BGA':
        if args.dataset == 'modelnet40':
            model = BGA(args, output_channels=40).to(device)
        elif args.dataset == 'ScanObjectNN':
            model = BGA(args, output_channels=15).to(device)
        else:
            raise Exception("Dataset Not supported")
    else:
        raise Exception("Not implemented")
    print(str(model))

    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-3)#
       

    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    #optimizer.load_state_dict(checkpoint['optimizer'])
    criterion = cal_loss
    best_test_acc = 0
   
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))#'2021-07-27_20-47'
    if args.log_dir is None:
        experiment_dir = args.exp_name + '/'+ timestr + '/'
    else:
        experiment_dir = args.exp_name + '/'+ args.log_dir + '/'
    #experiment_dir.mkdir(exist_ok=True)
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+ args.exp_name):
        os.makedirs('checkpoints/'+ args.exp_name)
    if not os.path.exists('checkpoints/'+ experiment_dir):
        os.makedirs('checkpoints/'+ experiment_dir) 
    if not os.path.exists('checkpoints/'+ experiment_dir +'/'+ 'logs'):
        log_dir = 'checkpoints/'+ experiment_dir +'/'+ 'logs'
        os.makedirs(log_dir)
    if not os.path.exists('checkpoints/'+ experiment_dir +'/'+ args.model):
        os.makedirs('checkpoints/'+ experiment_dir +'/'+ args.model)
    os.system('cp main.py checkpoints'+'/'+experiment_dir+'/'+'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + experiment_dir + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + experiment_dir + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + experiment_dir + '/' + 'data.py.backup')
    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % ('checkpoints/'+ args.exp_name + '/'+ timestr  +'/'+ 'logs', args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('PARAMETER ...')
    logger.info(args)
    logger.info('Start training...')

        ####################
    for epoch in range(args.epochs):
        scheduler.step()
        # Train
        ####################
        train_loss = 0.0
        segloss = 0.0
        count = 0.0
        contrast_loss = 0.0
        model.train()
        train_pred = []
        train_true = []
        full_globalfeat = torch.zeros(12,2048)
        truesaentropy = []
        wrongsaentropy = []
        weight = torch.ones(15,15)
        
        for data, label,_ in train_loader:
            data, label = data.to(device), label.to(device).squeeze() #data:[b,3,1024] label:[b]
            #mask = mask.to(device)#[24,1024]
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()  
            logits ,global_feature = model(data) #logits:[24,15],global_feature:[24,1024]             
            loss_cls = criterion(logits, label)
            #segloss = seg_loss(seg_pred,mask)
            global_feature = F.normalize(global_feature, p=2, dim=1)#[12,2048]
            #full_globalfeat = full_globalfeat.to(device)
            #global_feat = 0.9 * full_globalfeat + 0.1 * global_feature #[b*N,1024]  #加完后[13,1024] [] 不能随便直接加，因为维度不一样        
            #global_feature = global_feature.to(device)
            #full_globalfeat = torch.cat((global_feature, full_globalfeat), dim = 0) #
            metric_criterion = entropyweightMetricLoss()          
            loss_metric = metric_criterion(global_feature,label,logits) 
            #loss_metric = loss_metric.to(device)
            loss =1 * loss_cls + 0.1 * loss_metric #+ 0.5 *  segloss
            #loss = 1 * loss_cls# + 0.1 * loss_metric #+ 0.1*  reg_loss        
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            #for i in range(batch_size):
            #  if preds[i] == label[i]:  
            #      truesaentropy.append(entropy[i])
            #  if preds[i] != label[i]:
            #      wrongsaentropy.append(entropy[i]) 
            count += batch_size
            train_loss += loss.item() * batch_size
            #contrast_loss += loss_metric.item() * batch_size
            #segloss += loss_seg.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
            #mask = mask.view(-1)
            #seg_pred = seg_pred.view(-1)
            #train_pred.append(seg_pred.detach().cpu().numpy())
            #outstr = 'Train Iter %d, cls loss mean: %.6f, contrast loss mean: %.6f ,overall loss: %.6f' % (count//batch_size,loss_cls.item(),loss_metric.item(),loss.item())
            #outstr = 'Train Iter %d,cls loss mean: %.6f, contrast loss mean: %.6f, overall loss: %.6f ' % (count//batch_size,loss_cls.item(),loss_metric.item(),loss.item())
            outstr = 'Train Iter %d,cls loss mean: %.6f,overall loss: %.6f ' % (count//batch_size,loss_cls.item(),loss.item())
            if count%(batch_size*10) == 0:
                print(outstr)
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        
        #full_globalfeat = np.concatenate(full_globalfeat)

        outstr1 = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss*1.0/count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        print(outstr1)
        logger.info(outstr1)
        
        #print('result')
        #truesaentropy = np.array(truesaentropy)
        #wrongsaentropy = np.array(wrongsaentropy)
        #print('turemean %d,truestd %d,turemin %d,turemax %d' % (truesaentropy.mean(),truesaentropy.std(),truesaentropy.min(),truesaentropy.max()))
        #print('wrongmean %d,wrongstd %d,wrongmin %d,wrongmax %d' % (wrongsaentropy.mean(),wrongsaentropy.std(),wrongsaentropy.min(),wrongsaentropy.max()))


        #filename1 = str(epoch)+ 'training_globalfeat'
        #np.save('Dataanalysis/BGAinfoNCE/'+ str(filename1),full_globalfeat)
        #weight = Findweightofclass(full_globalfeat,train_true)



        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label  in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits,_ ,_= model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss*1.0/count,
                                                                              test_acc,
                                                                              avg_per_class_acc)
        print(outstr)
        logger.info(outstr)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'checkpoints/%s/%s/%s/model.t7' % (args.exp_name,timestr,args.model))
            #torch.save(model.state_dict(), 'checkpoints/'+ args.exp_name + '/'+ timestr +'/'+ args.model +'/'+model.t7)
        outstr = 'Current Best: %.6f' % best_test_acc
        print(outstr)
        logger.info(outstr)
    logger.info('End of training...')

           
def test(args, io):
    if args.dataset == 'modelnet40':
        test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                                 batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    elif args.dataset == 'ScanObjectNN':
        test_loader = DataLoader(ScanObjectNN(partition='test', num_points=args.num_points), num_workers=8,
                                 batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    else:
        raise Exception("Dataset Not supported")

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'pointnet':
        if args.dataset == 'modelnet40':
            model = PointNet(args, output_channels=40).to(device)
        elif args.dataset == 'ScanObjectNN':
            model = PointNet(args, output_channels=15).to(device)
        else:
            raise Exception("Dataset Not supported")
    elif args.model == 'dgcnn':
        if args.dataset == 'modelnet40':
            model = DGCNN(args, output_channels=40).to(device)
        elif args.dataset == 'ScanObjectNN':
            model = DGCNN(args, output_channels=15).to(device)
        else:
            raise Exception("Dataset Not supported")
    elif args.model == 'gbnet':
        if args.dataset == 'modelnet40':
            model = GBNet(args, output_channels=40).to(device)
        elif args.dataset == 'ScanObjectNN':
            model = GBNet(args, output_channels=15).to(device)
        else:
            raise Exception("Dataset Not supported")
    elif args.model == 'BGA':
        if args.dataset == 'modelnet40':
            model = BGA(args, output_channels=40).to(device)
        elif args.dataset == 'ScanObjectNN':
            model = BGA(args, output_channels=15).to(device)
        else:
            raise Exception("Dataset Not supported")
    else:
        raise Exception("Not implemented")
    print(str(model))
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    criterion = cal_loss
    full_globalfeat = []
    full_label = []
    entropy_list = []
    center_feat = []
    for data, label,mask in test_loader:

        data, label = data.to(device), label.to(device).squeeze()#label:[b]
        mask = mask.to(device)#[24,1024]
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        #logits,_ = model(data)
        #loss = criterion(logits, label)
        logits,global_feature = model(data) #logits:[24,15],global_feature:[24,1,1024]
            
        ##segment loss
        metric_criterion = modelnetyweightMetricLoss()
        global_feature = F.normalize(global_feature, p=2, dim=1)
        loss_metric, center_feat = metric_criterion(global_feature,label,logits) #global feature contrast

        ##criterion_ms = MultiSimilarityLoss()
        ##loss_metric = criterion_ms(global_feature,label)
        #loss_cls = criterion(logits, label)
        #loss = 0.1 * loss_metric+ 1 * loss_cls #+ 0.5 *  loss_seg

        #entropy_list.append(entropy)
        center_feat.append(center_feat)
        #full_globalfeat.append(global_feature.detach().cpu().numpy())
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
        #samplecount += 1
    
    
    center_feat_ = np.concatenate(center_feat)
    filename0 = 'center_feat'
    np.save('Dataanalysis/BGAinfoNCE/'+ str(filename0),center_feat)

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    #entropy_list = np.concatenate(entropy_list)
    #full_globalfeat = np.concatenate(full_globalfeat)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    print(outstr)

    #filename1 = 'full_gfeataa'
    #np.save('Dataanalysis/GBNET/'+ str(filename1),full_globalfeat)
    #filename2 = 'full_label'
    #np.save('Dataanalysis/GBNET/'+ str(filename2),test_true)
    #print('success')
    #filename3 = 'full_entropy'
    #np.save('Dataanalysis/GBNET/'+ str(filename3),entropy_list)




if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='gbnet_scanobjectnn', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='gbnet', metavar='N',
                        choices=['pointnet', 'dgcnn', 'gbnet','BGA','ContrastNet'],
                        help='Model to use')
    parser.add_argument('--dataset', type=str, default='ScanObjectNN', metavar='N',
                        choices=['modelnet40', 'ScanObjectNN'])
    parser.add_argument('--batch_size', type=int, default=12, metavar='batch_size',
                        help='Size of batch')
    parser.add_argument('--test_batch_size', type=int, default=8, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='checkpoints/gbnet_scanobjectnn-old/82.85model.t7', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    args = parser.parse_args()

    #_init_()



    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    print(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        print(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        print('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
