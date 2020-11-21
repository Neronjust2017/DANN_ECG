import random
import os
import sys
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
import numpy as np
from data_loader import GetDataset
from torchvision import datasets
from torchvision import transforms
from model import CNNModel
from test import test, valid
import argparse
from util import write_log
from tensorboardX import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def main(args):

    cuda = True
    cudnn.benchmark = True
    # data_root = '/home/weiyuhua/Challenge2020/Data/DG'
    data_root = '/home/yin/code/weiyuhua/Challenge2020/Data/DG'

    model_root = args.model_root
    logs = args.logs
    lr = args.lr
    batch_size = args.batch_size
    n_epoch = args.n_epoch
    unseen_index = args.unseen_index
    val_split = args.val_split

    manual_seed = random.randint(1, 10000)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    tb_dir = os.path.join(logs, 'tb_dir')
    if not os.path.exists(logs):
        os.makedirs(logs)
    if not os.path.exists(model_root):
        os.makedirs(model_root)
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)

    # Tensorboard
    train_writer = SummaryWriter(tb_dir + '/train')
    val_writer = SummaryWriter(tb_dir + '/valid')
    test_writer = SummaryWriter(tb_dir + '/test')

    # get train, val and test datasets
    D = GetDataset(data_root, unseen_index, val_split)
    train_datasets, val_datasets, test_dataset = D.get_datasets()

    # get dataloaders
    train_dataloaders = []
    for train_dataset in train_datasets:
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8
        )
        train_dataloaders.append(train_dataloader)

    val_dataloaders = []
    for val_dataset in val_datasets:
        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8
        )
        val_dataloaders.append(val_dataloader)

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )

    # load model
    my_net = CNNModel()

    # setup optimizer

    optimizer = optim.Adam(my_net.parameters(), lr=lr)

    loss_class = torch.nn.NLLLoss()
    loss_domain = torch.nn.NLLLoss()

    if cuda:
        my_net = my_net.cuda()
        loss_class = loss_class.cuda()
        loss_domain = loss_domain.cuda()

    for p in my_net.parameters():
        p.requires_grad = True

    # training
    best_accu_val = 0.0
    for epoch in range(n_epoch):

        len_dataloader = np.min(np.array([len(train_dataloaders[i]) for i in range(len(train_dataloaders))]))

        data_train_iters = []
        for train_dataloader in train_dataloaders:
            data_train_iter = iter(train_dataloader)
            data_train_iters.append(data_train_iter)

        for i in range(len_dataloader):

            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            err_label_s = []
            err_domain_s = []

            # err_label_all = torch.tensor(0.0)
            # err_domain_all = torch.tensor(0.0)
            err_label_all = 0
            err_domain_all = 0

            # training model using multi-source data
            for j, data_train_iter in enumerate(data_train_iters):
                data_train = data_train_iter.next()
                s_ecg, s_label = data_train

                my_net.zero_grad()
                batch_size = len(s_label)

                domain_label = (torch.ones(batch_size)*j).long()

                if cuda:
                    s_ecg = s_ecg.cuda()
                    s_label = s_label.cuda()
                    domain_label = domain_label.cuda()

                class_output, domain_output = my_net(input_data=s_ecg, alpha=alpha)
                err_label = loss_class(class_output, s_label)
                err_domain = loss_domain(domain_output, domain_label)

                err_label_s.append(err_label.data.cpu().numpy())
                err_domain_s.append(err_domain.data.cpu().numpy())
                err_label_all += err_label
                err_domain_all += err_domain

            # err = err_domain_all + err_label_all
            err = err_label_all
            err.backward()
            optimizer.step()

            print('\n')

            for j in range(len(train_dataloaders)):
                print('\r epoch: %d, [iter: %d / all %d], domain: %d, err_label: %f, err_domain: %f' \
                      % (epoch, i + 1, len_dataloader, j + 1, err_label_s[j], err_domain_s[j]))
                # tb training
                train_writer.add_scalar('err_label_%d' % (j), err_label_s[j])
                train_writer.add_scalar('err_domain_%d' % (j), err_domain_s[j])

            torch.save(my_net, '{0}/model_epoch_current.pth'.format(model_root))

        print('\n')

        ## validation
        val_accus, best_accu_val, val_err_label_s, val_err_domain_s = valid(val_dataloaders, model_root, best_accu_val)

        for i in range(len(val_dataloaders)):
            print('\r epoch: %d, Validation, domain: %d, accu: %f' % (epoch,  i + 1, val_accus[i]))
            # tb validation
            val_writer.add_scalar('err_label_%d' % (i), val_err_label_s[i])
            val_writer.add_scalar('err_domain_%d' % (i), val_err_domain_s[i])
            val_writer.add_scalar('accu_%d' % (i), val_accus[i])

        ## test
        test_accu, test_err_label = test(test_dataloader, model_root, model_best=False)
        test_writer.add_scalar('accu', test_accu)
        test_writer.add_scalar('err_label', test_err_label)


    result_path = os.path.join(logs, 'results.txt')
    print('============ Summary ============= \n')
    for i, train_dataloader in enumerate(train_dataloaders):
        train_accu, train_err_label = test(train_dataloader, model_root)
        write_log('Accuracy of the train dataset %d : %f err_label : %f' % (i+1, train_accu, train_err_label), result_path)

    for i, val_dataloader in enumerate(val_dataloaders):
        val_accu, val_err_label = test(val_dataloader, model_root)
        write_log('Accuracy of the val dataset %d : %f err_label : %f' % (i+1, val_accu, val_err_label), result_path)

    test_accu, test_err_label = test(test_dataloader, model_root)
    write_log('Accuracy of the test dataset %d : %f err_label : %f' % (i+1, test_accu, test_err_label), result_path)


if __name__ == "__main__":

    train_arg_parser = argparse.ArgumentParser(description="parser")

    train_arg_parser.add_argument("--model_root", type=str, default='models')
    train_arg_parser.add_argument("--batch_size", type=int, default=128,
                                  help="batch size for training, default is 64")
    train_arg_parser.add_argument("--n_epoch", type=int, default=100,
                                  help="number of epoches")
    train_arg_parser.add_argument("--num_classes", type=int, default=2,
                                  help="number of classes")
    train_arg_parser.add_argument("--unseen_index", type=int, default=0,
                                  help="index of unseen domain")
    train_arg_parser.add_argument("--lr", type=float, default=0.001,
                                  help='learning rate of the model')
    train_arg_parser.add_argument("--logs", type=str, default='logs/',
                                  help='logs folder to write log')
    train_arg_parser.add_argument("--val_split", type=float, default=0.9,
                                  help='validation split ratio')
    args = train_arg_parser.parse_args()

    main(args)