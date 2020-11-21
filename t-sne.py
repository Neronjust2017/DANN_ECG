import os
import numpy as np
import torch
import pickle as dill
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn import manifold

from model import CNNModel

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def load_data(root_folder):

    with open(os.path.join(root_folder, 'af_normal_data_processed.pkl'), 'rb') as file:
        data = dill.load(file)

    datasets = ['CSPC_data', 'PTB_XL_data', 'G12EC_data', 'Challenge2017_data']

    datas = []

    for source in datasets:
        af_data, normal_data = data[source]
        all_data = np.concatenate((af_data, normal_data), axis=0)
        all_label = np.zeros((len(all_data),))
        all_label[len(af_data):] = 1

        all_data = all_data.swapaxes(1,2)
        datas.append([all_data, all_label])

    return datas


if __name__ == '__main__':

    run_ids = ['1', '2', '3', '4']
    log_ids = ['0', '1', '2', '3']

    data_path = '/home/weiyuhua/Challenge2020/Data/DG'

    # for name, m in network.named_modules():
    #     print(name)
    # conv1 conv2 conv3 pooling fc1 fc2 fc3 bn1 bn2 bn3 dropout act

    # data
    datas = load_data(data_path)
    # unseen_data = datas[unseen_index]
    # del datas[unseen_index]

    # gpu
    cuda = True
    device = 'cpu'
    if cuda and torch.cuda.is_available():
        device = 'cuda'


    # Inferance


    for run_id in run_ids:
        for log_id in log_ids:

            features_all = []
            labels_all = []
            d_labels_all = []

            run_path = 'run_' + run_id
            model_path = run_path+'/models_' + log_id

            # model
            network = torch.load(os.path.join(model_path, 'model_epoch_best.pth'))
            network.eval()

            for i, data in enumerate(datas):
                test_ecgs = data[0]
                test_labels = data[1]
                test_d_labels = np.ones(shape=(len(test_labels, ))) * i
                threshold = 1000
                n_slices_test = int(len(test_ecgs) / threshold) + 1
                indices_test = []
                for per_slice in range(n_slices_test - 1):
                    indices_test.append(int(len(test_ecgs) * (per_slice + 1) / n_slices_test))
                test_ecg_splits = np.split(test_ecgs, indices_or_sections=indices_test)

                # Verify the splits are correct
                test_ecg_splits_2_whole = np.concatenate(test_ecg_splits)
                assert np.all(test_ecgs == test_ecg_splits_2_whole)

                # split the test data into splits and test them one by one

                features = []
                network.eval()
                for test_ecg_split in test_ecg_splits:
                    ecgs_test = Variable(torch.from_numpy(np.array(test_ecg_split, dtype=np.float32))).to(device)

                    # feature
                    out = ecgs_test
                    # for layer in list(network.feature):
                    #     out = layer(out)
                    # out = torch.mean(out, dim=2)

                    for layer in list(network.feature):
                        out = layer(out)

                    out = torch.mean(out, dim=2)

                    for layer in list(network.class_classifier):
                        out = layer(out)
                        if isinstance(layer, torch.nn.ReLU):
                            out = layer(out)
                            break

                    out = out.cpu().data.numpy()
                    features.append(out)

                features = np.concatenate(features, axis=0)
                features_all.append(features)
                labels_all.append(test_labels)
                d_labels_all.append(test_d_labels)

            features_all = np.concatenate(features_all, axis=0)
            labels_all = np.concatenate(labels_all, axis=0)
            d_labels_all = np.concatenate(d_labels_all, axis=0)

            # T-SNE
            embs = features_all
            tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
            X_tsne = tsne.fit_transform(embs)

            print("Org data dimension is {}.Embedded data dimension is {}".format(embs.shape[-1], X_tsne.shape[-1]))

            '''嵌入空间可视化'''
            # colors = ['blue', 'cyan', 'green', 'red', 'yellow', 'magenta', 'black']
            # colors = ['limegreen', 'tomato', 'purple', 'yellow', 'orange', 'royalblue', 'black']
            colors = ['red', 'royalblue', 'limegreen', 'darkorange']

            x_min, x_max = X_tsne.min(0), X_tsne.max(0)
            X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
            plt.figure(figsize=(16, 16))

            domain = ['CSPC', 'PTB_XL', 'G12EC', 'Challenge2017']

            for i in range(X_norm.shape[0]):

                if labels_all[i] == 0:
                    plt.text(X_norm[i, 0], X_norm[i, 1], '*', color=colors[int(d_labels_all[i])],
                         fontdict={'weight': 'bold', 'size': 9}, label=domain[int(d_labels_all[i])])
                else:
                    plt.text(X_norm[i, 0], X_norm[i, 1], 'o', color=colors[int(d_labels_all[i])],
                             fontdict={'weight': 'bold', 'size': 9}, label=domain[int(d_labels_all[i])])

            plt.xticks([])
            plt.yticks([])

            plt.legend(labels=domain)

            title = 'run_%s_unseen_%s_baseline_classifier_relu1' % (run_id, log_id)
            plt.title(title)

            plt.savefig(title + '.png')
            plt.show()
            plt.cla()

            # for i in range(24):
            #     for j in range(X_norm.shape[0]):
            #         if labels[j][i] == True:
            #              plt.text(X_norm[j, 0], X_norm[j, 1], str(domains[j]), color=colors[int(domains[j])]),
            #              fontdict={'weight': 'bold', 'size': 9}
            #     plt.xticks([])
            #     plt.yticks([])
            #     plt.title(classes[i])
            #     plt.savefig('t-sne/model-0/' + classes[i] + '.png')
            #     plt.show()
            #     plt.cla()






