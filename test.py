import os
import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms
from torchvision import datasets

loss_class = torch.nn.NLLLoss()
loss_domain = torch.nn.NLLLoss()

def test(test_dataloader, model_root, model_best=True):
    cuda = True
    alpha = 0

    """ test """

    if model_best:
        my_net = torch.load(os.path.join(
            model_root, 'model_epoch_best.pth'
        ))
    else:
        my_net = torch.load(os.path.join(
            model_root, 'model_epoch_current.pth'
        ))

    my_net = my_net.eval()

    if cuda:
        my_net = my_net.cuda()

    len_dataloader = len(test_dataloader)
    data_target_iter = iter(test_dataloader)

    i = 0
    n_total = 0
    n_correct = 0
    err_label_t = 0

    while i < len_dataloader:

        # test model using target data
        data_target = data_target_iter.next()
        t_ecg, t_label = data_target

        batch_size = len(t_label)

        if cuda:
            t_ecg = t_ecg.cuda()
            t_label = t_label.cuda()

        class_output, _ = my_net(input_data=t_ecg, alpha=alpha)
        err_label = loss_class(class_output, t_label)
        err_label_t += err_label.data.cpu().numpy()

        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(t_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        i += 1

    accu = n_correct.data.numpy() * 1.0 / n_total
    err_label_t /= len_dataloader

    return accu, err_label_t

def valid(val_dataloaders, model_root, best_accu):
    cuda = True
    alpha = 0

    """ validation """
    my_net = torch.load(os.path.join(
        model_root, 'model_epoch_current.pth'
    ))

    my_net = my_net.eval()

    if cuda:
        my_net = my_net.cuda()

    accus = []
    err_label_s = []
    err_domain_s = []

    for j, val_dataloader in enumerate(val_dataloaders):

        len_dataloader = len(val_dataloader)
        data_val_iter = iter(val_dataloader)

        i = 0
        n_total = 0
        n_correct = 0

        err_label_j = 0
        err_domain_j = 0

        while i < len_dataloader:

            # test model using valid data
            data_target = data_val_iter.next()
            t_ecg, t_label = data_target

            batch_size = len(t_label)
            domain_label = (torch.ones(batch_size) * j).long()

            if cuda:
                t_ecg = t_ecg.cuda()
                t_label = t_label.cuda()
                domain_label = domain_label.cuda()

            class_output, domain_output = my_net(input_data=t_ecg, alpha=alpha)
            err_label = loss_class(class_output, t_label)
            err_domain = loss_domain(domain_output, domain_label)

            err_label_j += err_label.data.cpu().numpy()
            err_domain_j += err_domain.data.cpu().numpy()

            pred = class_output.data.max(1, keepdim=True)[1]
            n_correct += pred.eq(t_label.data.view_as(pred)).cpu().sum()
            n_total += batch_size

            i += 1

        accu = n_correct.data.numpy() * 1.0 / n_total
        err_label_s.append(err_label_j / len_dataloader)
        err_domain_s.append(err_domain_j / len_dataloader)
        accus.append(accu)

    mean_accu = np.mean(accus)

    if mean_accu > best_accu:
        best_accu = mean_accu
        torch.save(my_net, '{0}/model_epoch_best.pth'.format(model_root))

    return accus, best_accu, err_label_s, err_domain_s