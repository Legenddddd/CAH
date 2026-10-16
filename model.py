import torch.nn as nn
import torch
import torch.nn.functional as F

import math
from torch.nn.parameter import Parameter
from torch.nn import init
from skimage import measure

from PIL import Image
from torchvision import transforms
import random as rand

class Model(nn.Module):
    def __init__(self, model, feature_size, num_ftrs, components_num, topn, bit, alpha):
        super(Model, self).__init__()

        self.backbone = model
        self.num_ftrs = num_ftrs
        self.topn = topn
        self.im_sz = 224
        self.pad_side = 112
        self.class_num = components_num
        self.lamda = alpha
        print(self.topn)

        self.pool3 = nn.AvgPool2d((4, 4), stride=(4, 4))
        self.pool2 = nn.AvgPool2d((2, 2), stride=(2, 2))

        self.Nsub_conv3 = nn.Sequential(
            BasicConv(self.num_ftrs//4, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs//2, kernel_size=3, stride=1, padding=1, relu=True),
            nn.AdaptiveMaxPool2d(1))
        self.Nsub_fc3 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs//2),
            nn.Linear(self.num_ftrs//2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Dropout(0.5))
        self.Ncls_3 = nn.Sequential(
            nn.Linear(feature_size, components_num))

        #Output layer
        self.Nsub_conv2 = nn.Sequential(
            BasicConv(self.num_ftrs//2, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs//2, kernel_size=3, stride=1, padding=1, relu=True),
            nn.AdaptiveMaxPool2d(1))
        self.Nsub_fc2 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs//2),
            nn.Linear(self.num_ftrs//2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Dropout(0.5))

        self.Nsub_conv1 = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs//2, kernel_size=3, stride=1, padding=1, relu=True),
            nn.AdaptiveMaxPool2d(1))
        self.Nsub_fc1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs//2),
            nn.Linear(self.num_ftrs//2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Dropout(0.5))
        self.Ncls_1 = nn.Sequential(
            nn.Linear(feature_size, components_num))
        
        # concat features from different stages
        self.Ncon = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs//2 * 3),
            nn.Linear(self.num_ftrs//2 * 3, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
        )

        self.hash = nn.Linear(feature_size, bit, bias=False)
        init.kaiming_uniform_(self.hash.weight, a=math.sqrt(5))
        self.hash_layer = Parameter(torch.Tensor(bit, components_num))
        init.kaiming_uniform_(self.hash_layer, a=math.sqrt(5))

        self.C_Match = C_Match(alpha=alpha, dim=feature_size, cN=components_num)

    def forward(self, x, index=None, pair_matrix=None, is_train=True):
        _, _, F3, F2, F1 = self.backbone(x)

        batch = x.shape[0]
        #Distinct critical image region localization
        F_stack = torch.cat((self.pool3(F3).detach(), self.pool2(F2).detach(), F1.detach()), 1).clone()
        F_stack = torch.sum(F_stack, dim=1, keepdim=False)
        patch_imgs = torch.zeros([batch, self.topn, 3, 112, 112]).to(x.device)
        for i in range(batch):
            Fmap = F_stack[i,:,:]
            coordinate = SCDA(Fmap, self.topn, 112)
            for j in range(self.topn):
                [x0, y0, x1, y1] = coordinate[j]
                patch_imgs[i:i + 1, j] = F.interpolate(x[i:i + 1, :, x0:(x1 + 1), y0:(y1 + 1)],
                                                            size=(112, 112),mode='bilinear',
                                                            align_corners=True)

        #Multi-region feature embedding
        patch_imgs = patch_imgs.view(batch*self.topn, 3, 112, 112)
        _, _, F3_part, F2_part, F1_part = self.backbone(patch_imgs.detach())
        f3_patch = self.Nsub_conv3(F3_part).view(batch*self.topn, -1)
        f2_patch = self.Nsub_conv2(F2_part).view(batch*self.topn, -1)
        f1_patch = self.Nsub_conv1(F1_part).view(batch*self.topn, -1)
        yp3 = self.Ncls_3(self.Nsub_fc3(f3_patch))
        yp2, _ = self.C_Match(self.Nsub_fc2(f2_patch))
        yp1 = self.Ncls_1(self.Nsub_fc1(f1_patch))
        yp0, _ = self.C_Match(self.Ncon(torch.cat((f3_patch, f2_patch, f1_patch), -1)))

        #Cross-layer semantic information transfer
        f3 = self.Nsub_conv3(F3).view(batch, -1)
        f2 = self.Nsub_conv2(F2).view(batch, -1)
        f1 = self.Nsub_conv1(F1).view(batch, -1)
        y3 = self.Ncls_3(self.Nsub_fc3(f3))
        if is_train:
            assign = self.C_Match(self.Nsub_fc2(f2), index, pair_matrix, assign=True)
        y2, M = self.C_Match(self.Nsub_fc2(f2))
        y1 = self.Ncls_1(self.Nsub_fc1(f1))
        y0, _ = self.C_Match(self.Ncon(torch.cat((f3, f2, f1), -1)))


        code = self.hash(M)

        codel = self.lamda * code

        codey = codel.tanh().matmul((self.lamda * self.hash_layer).tanh())

        if is_train:
            return codel, y1, y2, y3, y0, yp1, yp2, yp3, yp0, assign, codey
        else:
            return code, y1, y2, y3, y0, yp1, yp2, yp3, yp0

    
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class C_Match(nn.Module):
    def __init__(self, alpha=20, dim=512, cN=200, beta=0.9):
        super(C_Match, self).__init__()
        self.alpha = alpha
        self.cN = cN
        self.C_vector = Parameter(torch.Tensor(dim, cN))
        init.kaiming_uniform_(self.C_vector, a=math.sqrt(5))
        self.beta = Parameter(torch.Tensor([beta]))

        self.register_buffer('C_vector_index', torch.full((cN,),-1, dtype=torch.long))


    def forward(self, input, global_index = None, current_rows = None, assign = False):
        if assign:
            batch_size = input.size(0)
            device = input.device
            input = F.normalize(input, p=2, dim=1)
            centers = F.normalize(self.C_vector, p=2, dim=0)
            matching = input.matmul(centers)
            assignments = torch.zeros(batch_size, dtype=torch.long, device=device)
            for i in range(batch_size):
                positive = torch.nonzero(current_rows[i] == 1, as_tuple=True)[0].to(device)
                mask = self.C_vector_index.view(-1, 1) == positive.view(1, -1)
                if mask.any():
                    j = torch.nonzero(mask, as_tuple=True)[0][0]
                    assignments[i] = j
                else:
                    available_mask = self.C_vector_index == -1
                    if not available_mask.any():
                        raise RuntimeError("No available classes in B.")

                    masked_similarity = matching[i].clone()
                    masked_similarity[~available_mask] = -float('inf')
                    j = torch.argmax(masked_similarity)

                    self.C_vector_index[j] = global_index[i]
                    assignments[i] = j
            return assignments

            
        else:
            input = F.normalize(input, p=2, dim=1)
            centers = F.normalize(self.C_vector, p=2, dim=0)

            matching = input.matmul(centers)
            cls_vector = matching.reshape(-1, self.cN)

            (_, index) = torch.max(matching, dim=1)
            prototype = centers[:, index].detach().t()

            return self.alpha * cls_vector, self.beta * prototype +(1-self.beta) * input


def SCDA(fms, topn, size):
    A = fms
    a = torch.mean(A)
    H = (A > a).float()

    mask_np = H.cpu().numpy().reshape(7, 7)
    component_labels = measure.label(mask_np)

    properties = measure.regionprops(component_labels)
    areas = []
    for prop in properties:
        areas.append(prop.area)
    if len(areas) == 0:
        print('areas size = 0')
        print(A)
        print(a)

        return [1, 1, 223, 223]
    max_idx = areas.index(max(areas))
    bbox = properties[max_idx].bbox

    x_lefttop = bbox[0] * 32 - 1
    y_lefttop = bbox[1] * 32 - 1
    x_rightlow = bbox[2] * 32 - 1
    y_rightlow = bbox[3] * 32 - 1

    if x_lefttop < 0:
        x_lefttop = 0
    if y_lefttop < 0:
        y_lefttop = 0
    coordinate_SDCA = [x_lefttop, y_lefttop, x_rightlow, y_rightlow]
    coordinate = []
    coordinate.append(coordinate_SDCA)
    for i in range(topn-1):
        centerx = rand.randint(x_lefttop, x_rightlow)
        centery = rand.randint(y_lefttop, y_rightlow)
        rand_size_x = rand.randint(size/2*0.75, size/2*1.25)
        rand_size_y = rand.randint(size / 2 * 0.75, size / 2 * 1.25)
        x_lefttop_rand = centerx - rand_size_x
        x_rightlow_rand = centerx + rand_size_x
        y_lefttop_rand = centery - rand_size_y
        y_rightlow_rand = centery + rand_size_y
        if x_lefttop_rand < 0:
            x_lefttop_rand = 0
        if x_rightlow_rand > 223:
            x_rightlow_rand = 223
        if y_lefttop_rand < 0:
            y_lefttop_rand = 0
        if y_rightlow_rand > 223:
            y_rightlow_rand = 223
        coordinate.append([x_lefttop_rand, y_lefttop_rand, x_rightlow_rand, y_rightlow_rand])

    return coordinate


def save_image(image, file_name = '123'):
    img = image.detach().cpu().permute(1, 2, 0) * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
    img = img.permute(2, 0, 1)
    img = transforms.ToPILImage()(img)
    img.save(file_name)
