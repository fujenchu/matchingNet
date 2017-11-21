import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from nets import Net
from config import args
from dataloader import Producer
from utils import read_miniImageNet_pathonly
from Queue import Queue
import numpy as np
import threading
import os


def main(args):
    '''
    main function
    '''
    EPOCH_SIZE = args.num_episode*args.num_query*args.way_train
    EPOCH_SIZE_TEST = args.num_episode*args.num_query*args.way_test
    SM_CONSTANT = 50


    '''define network'''
    net = Net(args.num_in_channel, args.num_filter)
    if torch.cuda.is_available():
        net.cuda()


    '''
    load model if needed
    '''
    if args.model_load_path!='':
        net.load_state_dict(torch.load(args.model_load_path))
        net.cuda()
        print('model loaded')


    ''' define loss, optimizer'''
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)


    '''get data'''
    trainList = read_miniImageNet_pathonly(TESTMODE=False,
                                           miniImageNetPath='/dataset/miniImageNet_Ravi/',
                                           imgPerCls=600)
    testList = read_miniImageNet_pathonly(TESTMODE=True,
                                          miniImageNetPath='/dataset/miniImageNet_Ravi/',
                                          imgPerCls=600)
    queue = Queue(maxsize=3)
    producer = threading.Thread(target=Producer, args=(queue, trainList, args.batch_size, EPOCH_SIZE, "training"))
    producer.start()
    queue_test = Queue(maxsize=50)
    producer_test = threading.Thread(target=Producer,args=(queue_test, testList, args.batch_size, EPOCH_SIZE_TEST, "testing"))
    producer_test.start()


    ''' training'''
    for epoch in range(1000):
        running_loss = 0.0
        avg_accu_Train = 0.0
        avg_accu_Test = 0.0
        total_batch = int(EPOCH_SIZE / args.batch_size)
        total_batch_test = int(EPOCH_SIZE_TEST / args.batch_size)

        for i in range(total_batch):
            # get inputs
            batch = queue.get()
            labels = torch.from_numpy(batch[0])
            images = batch[1:]
            images_all = torch.from_numpy(np.transpose(np.concatenate(images),(0,3,1,2))).float()

            # wrap in Variable
            if torch.cuda.is_available():
                images_all, labels = Variable(images_all.cuda()), Variable(labels.cuda())
            else:
                images_all, labels = Variable(images_all), Variable(labels)

            # zero gradients
            optimizer.zero_grad()

            # forward + backward + optimizer
            feature_s_all_t0_p = net(images_all)
            #split_list = [args.batch_size for _ in range(args.way_train + 1)]
            feature_s_all_t0_p = torch.split(feature_s_all_t0_p, args.batch_size, 0)
            cosineDist_list = [[] for _ in range(args.way_train)]

            for idx in range(args.way_train):
                cosineDist_list[idx] = SM_CONSTANT * torch.sum(
                    torch.mul(feature_s_all_t0_p[-1].div(torch.norm(feature_s_all_t0_p[-1], p=2, dim=1, keepdim=True).expand_as(feature_s_all_t0_p[-1])),
                              feature_s_all_t0_p[idx].div(torch.norm(feature_s_all_t0_p[idx], p=2, dim=1, keepdim=True).expand_as(feature_s_all_t0_p[idx]))), dim=1, keepdim=True)
            cosineDist_all = torch.cat(cosineDist_list, 1)

            labels = labels.squeeze(1)
            loss = criterion(cosineDist_all, labels)
            loss.backward()
            optimizer.step()

            # summing up
            running_loss += loss.data[0]
            _, predicted = torch.max(cosineDist_all.data, 1)
            avg_accu_Train += (predicted == labels.data).sum()
            if i % 1000 == 999:
                print('[%d, %5d] train loss: %.3f  train accuracy: %.3f' % (epoch + 1, i + 1, running_loss / 1000, avg_accu_Train/(1000*args.batch_size)))
                running_loss = 0.0
                avg_accu_Train = 0.0

            if (i+1) % args.save_step == 0:
                torch.save(net.state_dict(),
                           os.path.join(args.model_path,
                                        'model-%d-%d.pkl' %(epoch+1, i+1)))
        net.eval()
        for i in range(total_batch_test):
            # get inputs
            batch = queue_test.get()
            labels = torch.from_numpy(batch[0])
            images = batch[1:]
            images_all = torch.from_numpy(np.transpose(np.concatenate(images), (0, 3, 1, 2))).float()

            # wrap in Variable
            if torch.cuda.is_available():
                images_all, labels = Variable(images_all.cuda()), Variable(labels.cuda())
            else:
                images_all, labels = Variable(images_all), Variable(labels)

            # forward
            feature_s_all_t0_p = net(images_all)
            # split_list = [args.batch_size for _ in range(args.way_train + 1)]
            feature_s_all_t0_p = torch.split(feature_s_all_t0_p, args.batch_size, 0)
            cosineDist_list = [[] for _ in range(args.way_test)]

            for idx in range(args.way_test):
                cosineDist_list[idx] = SM_CONSTANT * torch.sum(
                    torch.mul(feature_s_all_t0_p[-1].div(torch.norm(feature_s_all_t0_p[-1], p=2, dim=1, keepdim=True).expand_as(feature_s_all_t0_p[-1])),
                              feature_s_all_t0_p[idx].div(torch.norm(feature_s_all_t0_p[idx], p=2, dim=1, keepdim=True).expand_as(feature_s_all_t0_p[idx]))), dim=1, keepdim=True)

            cosineDist_all = torch.cat(cosineDist_list, 1)
            _, predicted = torch.max(cosineDist_all.data, 1)
            labels = labels.squeeze(1)
            avg_accu_Test += (predicted == labels.data).sum()

        print('test accuracy: %.3f' % (avg_accu_Test/(total_batch_test*args.batch_size)))
        avg_accu_Test = 0.0


if __name__ == '__main__':
    main(args)