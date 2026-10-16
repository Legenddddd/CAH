from __future__ import print_function
import torch
import argparse
import torch.nn.functional as F
import time
from data.data_loader import load_data
from utils import *
import os

def train():
    parser = argparse.ArgumentParser('FGIR', add_help=False)
    parser.add_argument('--epochs', type=int, default=100, help="training epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="batch size")
    parser.add_argument('--dataset_name', type=str, default="cub", help="dataset name")
    parser.add_argument('--M', type=int, default=4, help="parts number")
    parser.add_argument('--backbone', type=str, default="resnet50", help="backbone")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--alpha', type=int, default=12, help="alpha")
    parser.add_argument('--gamma', type=float, default=0.25, help="gamma")
    parser.add_argument('--GPU', type=str, default="12", help="1,2")
    parser.add_argument('--bit', type=int, default=32, help="bit")
    parser.add_argument('--WD', type=float, default=5e-4, help="weight_decay")
    parser.add_argument('--momentum', type=float, default=0.9, help="momentum")

    args, _ = parser.parse_known_args()
    print(args)

    os.environ['CUDA_DEVICE_ORDRE'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU

    epochs = args.epochs
    batch_size = args.batch_size
    hash_bit = args.bit
    num_workers = 16

    ## Data
    data_config = {"cub": [200, "/data2/zhaolijun/Datasets/CUB_200_2011"],
                   "air": [100, "/data2/zhaolijun/Datasets/fgvc-aircraft-2013b"],
                   "food101": [101, "/data2/zhaolijun/Datasets/food-101/"],
                   "nabirds": [555, "/data2/zhaolijun/Datasets/nabirds/"],
                   "vegfru": [292, "/data2/zhaolijun/Datasets/vegfru/"],
                   }
    dataset_name = args.dataset_name
    classes_num, data_root = data_config[dataset_name]
    testloader, trainloader, datasetloader, num_test, num_train = load_data(
        dataset_name,
        data_root,
        -1,
        -1,
        batch_size,
        num_workers,
    )

    num_selected = int(num_train*1.0)
    print('num_test:', num_test, '| num_train&dataset:', num_train, '| num_selected:', num_selected)

    time_str = time.strftime("%Y%m%d-%H%M%S")
    log_file = open(args.dataset_name +'_topN_' + str(args.M)+ '_'+ str(args.gamma) + '_'+ str(args.alpha)+ '_'+ str(hash_bit) + '_' + time_str + '.log', 'a')
    log_file.write(str(args))

    print(len(trainloader), len(testloader), len(datasetloader))

  ##############################
    train_label_oh, pair_matrix  = dataset_label(datasetloader, classes_num)
    components_num = count_components(pair_matrix)
    print('n:',components_num)
    ## Model
    net = load_model(backbone=args.backbone, pretrain=True, require_grad=True, components_num=components_num, topn=args.M,
                         bit=hash_bit, alpha=args.alpha)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        net = net.to(device)
        netp = torch.nn.DataParallel(net)
    else:
        device = torch.device('cpu')
        netp = net

    deep_paras = [para for name, para in net.named_parameters() if "backbone" not in name]
    optimizer = torch.optim.SGD(
        [{'params': deep_paras},
         {'params': net.backbone.parameters(), 'lr': args.lr / 10.0}],
        lr=args.lr, momentum=0.9, weight_decay=5e-4)
    ## Loss
    CELoss = torch.nn.CrossEntropyLoss()


    #Param ini for loss

    B = torch.zeros((num_train, hash_bit)).to(device)
    B_code = torch.zeros((num_train, hash_bit)).to(device)
    B_label = torch.tensor(train_label_oh).to(device).float()

    for epoch in range(1, epochs + 1):
        # print('\nEpoch: %d' % epoch)
        if epoch > 70:
            optimizer.param_groups[0]['lr'] = args.lr * 0.1
            optimizer.param_groups[1]['lr'] = (args.lr / 10.0) * 0.1



        net.train()
        selected_index = torch.randperm(num_train)
        selected_index = selected_index[0:num_selected]
        B_code_temp = B_code[selected_index]

        for _, (index, inputs, targets) in enumerate(trainloader):
            if inputs.shape[0] < batch_size:
                continue
            if torch.cuda.is_available():
                inputs, targets = inputs.to(device), targets.to(device)

            # forward
            optimizer.zero_grad()

            current_rows = pair_matrix[index]
            code, _, y2, _, y4, _, yp2, _, yp4, assign, codey = netp(inputs,index,current_rows)
            code = code.tanh()


            current_rows = current_rows.float()
            r = current_rows.sum() / (1 - current_rows).sum()
            S_B = current_rows* (1 + r) - r
            pair_loss = ((args.bit * S_B.cuda() - code @ B_code_temp.t()) ** 2).sum() / (code.shape[0] * B_code_temp.shape[0])
            center_loss = CELoss(codey, assign)

            loss_hash = 2 * (center_loss + args.gamma*pair_loss)

            loss2 = CELoss(y2, assign)
            loss4 = CELoss(y4, assign)
            loss_cls = loss2 + loss4

            assign_parts = assign.unsqueeze(1).repeat(1, args.M).view(-1)
            lossp2 = CELoss(yp2, assign_parts)
            lossp4 = CELoss(yp4, assign_parts)
            loss_region_cls = lossp2 + lossp4

            p, q = F.softmax(y2 + y4, dim=-1).detach(), F.log_softmax(y2, dim=-1)
            loss_trans = torch.mean(-torch.sum(p * q, dim=-1))
            p, q = F.softmax(yp2 + yp4, dim=-1).detach(), F.log_softmax(yp2, dim=-1)
            loss_region_trans = torch.mean(-torch.sum(p * q, dim=-1))


            B[index, :] = code.detach()

            loss = loss_hash + loss_cls + loss_trans + (loss_region_cls + loss_region_trans)

            # backward
            loss.backward()
            optimizer.step()

        #updata centers and hash codes
        C = (B_label.t()) @ B
        C = F.normalize(C / torch.sum(B_label.t(), dim=1, keepdim=True),dim=1)  # mean
        C = C.sign()
        B_code = B.sign()

        # test
        if epoch < 3 or epoch % 10 == 0 or epoch > 95:  #
            net.eval()

            dataset_code = torch.zeros(num_train, hash_bit).to(device)
            dataset_label_oh = torch.zeros(num_train, classes_num).to(device)
            test_code = torch.zeros(num_test, hash_bit).to(device)
            test_label_oh = torch.zeros(num_test, classes_num).to(device)

            with torch.no_grad():
                for ii, (index, inputs, targets) in enumerate(datasetloader):
                    label_oh = one_hot_label(targets, classes_num)
                    dataset_label_oh[index, :] = label_oh.to(device)

                    if torch.cuda.is_available():
                        inputs, targets = inputs.to(device), targets.to(device)
                    code, _, _, _, _, _, _, _, _ = net(inputs, is_train=False)

                    dataset_code[index, :] = torch.sign(code)
                for ii, (index, inputs, targets) in enumerate(testloader):
                    label_oh = one_hot_label(targets, classes_num)
                    test_label_oh[index, :] = label_oh.to(device)

                    if torch.cuda.is_available():
                        inputs, targets = inputs.to(device), targets.to(device)
                    code, y1, y2, y3, y4, _, _, _, _ = net(inputs, is_train=False)

                    test_code[index, :] = torch.sign(code)


            print(optimizer.param_groups[0]['lr'], test_code.size(), test_label_oh.size(), dataset_code.size(), dataset_label_oh.size(), num_train)


            mAP = calc_map_k(test_code, dataset_code, test_label_oh, dataset_label_oh)
            result_str_hash = 'Epoch %d | dataset length = %d | bit = %d | map = %.5f\n' % (
                epoch, dataset_code.shape[0], hash_bit, mAP)
            print(result_str_hash)
            log_file.write(result_str_hash)

    log_file.close()


if __name__ == "__main__":
    train()
