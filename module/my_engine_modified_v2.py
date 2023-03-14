from module.btrfly import BtrflyNet
from PIL import Image
from torchvision import transforms

import matplotlib.pyplot as plt
import torch, os, timeit
import numpy as np

def render_data(img_front, img_back):

    kupu = BtrflyNet()
    kupu.load_state_dict( torch.load('./model/model-btr-paper.pt', map_location=torch.device('cpu')) )

    tfms = transforms.Compose([transforms.ToTensor()])

    # imsize = (512, 128, 1)

    def masking(msk):
        tmp = msk.round()
        return [(tmp == i).astype('float').tolist() for i in range(13)]

    X, y = [], []

    X.append([
        tfms(Image.open(f'./upload/img/' + img_front)).tolist(),
        tfms(Image.open(f'./upload/img/' + img_back)).tolist()
    ])
    y.append([
        # masking(plt.imread(f'./data/msk/indo-101.png')[...,0] * 12),
        # masking(plt.imread(f'./data/msk/indo-102.png')[...,0] * 12)
    ])
    X_valid, y_valid = torch.Tensor(X), torch.Tensor(y)
    n_data_valid = len(X_valid)

    def masking_torch(msk):
        return torch.Tensor([ (msk == i).cpu().numpy().astype('float') for i in range(13) ])

    # inp_size = (1,  3, 512, 128)
    inp_size = (1, 3, 512, 128)
    # out_size = (1, 13, 512, 128)
    dsc_size = (1, 1, 13, 512, 128)

    # start = timeit.default_timer()

    for i in range(n_data_valid):
            
        x, y = X_valid[i], y_valid[i]

        out_ant, out_pos = kupu(
            x[0].reshape(inp_size),
            x[1].reshape(inp_size)
        )

        tmp = torch.cat([
            masking_torch(out_ant.argmax(axis=1)[0]).reshape(dsc_size),
            masking_torch(out_pos.argmax(axis=1)[0]).reshape(dsc_size)
        ], axis=1)

        if i == 0: y_predv = tmp + 0
        else: y_predv = torch.cat([y_predv, tmp], axis=0)

    # ini warna
    cp = {
        0 : [255, 255, 255],
        1 : [0, 0, 0],
        2 : [17,17,17],
        3 : [34,34,34],
        4 : [51,51,51],
        5 : [68,68,68],
        6 : [85,85,85],
        7 : [102,102,102],
        8 : [119,119,119],
        9 : [136,136,136],
        10 : [153,153,153],
        11 : [170,170,170],
        12 : [187,187,187],
        # 13 : [204,204,204],
        # 14 : [221,221,221],
        # 15 : [238,238,238],
    }

    # pewarnaan disini
    def map_clr(mask):
        res = []
        for row in mask:
            new_row = [cp[x] for x in row]
            res.append(new_row)
        return np.array(res)

    n = 4

    np.random.seed(76)

    vl_idx = np.random.choice(range(n_data_valid), n)

    # img size
    # plt.figure(figsize=(2, 5))
    fig = plt.figure(frameon=False)
    fig.set_figwidth(2)
    fig.set_figheight(5)

    # this for disabling the bg
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    # depan
    # plt.subplot(2, n*2, (2*i)+1)
    # ax.imshow(X_valid[vl_idx[i]][0].permute(1, 2, 0))
    ax.imshow(map_clr(y_predv[vl_idx[i]][0].argmax(axis=0).numpy()), alpha=1)

    plt.savefig('./static/coba.png')

    # belakang
    # plt.subplot(2, n*2, (2*i)+2)
    ax.imshow(X_valid[vl_idx[i]][1].permute(1, 2, 0))
    ax.imshow(map_clr(y_predv[vl_idx[i]][1].argmax(axis=0).numpy()), alpha=0.5)

    plt.savefig('./static/img_back_result.png')