from module.btrfly import BtrflyNet
from PIL import Image
from torchvision import transforms
from os.path import exists

import matplotlib.pyplot as plt
import torch
import numpy as np

def render_data(img_front, img_back, render_type):

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

    # ini warna (r,g,b)
    if render_type == "normal":
        cp = {
            0 : [1.0       , 1.0       , 1.0       ],
            1 : [0.6901961 , 0.9019608 , 0.05098039],
            2 : [0.        , 0.5921569 , 0.85882354],
            3 : [0.49411765, 0.9019608 , 0.8862745 ],
            4 : [0.6509804 , 0.21568628, 0.654902  ],
            5 : [0.9019608 , 0.6156863 , 0.7058824 ],
            6 : [0.654902  , 0.43137255, 0.3019608 ],
            7 : [0.47843137, 0.        , 0.09411765],
            8 : [0.22352941, 0.25490198, 0.72156864],
            9 : [0.9019608 , 0.85490197, 0.        ],
            10 : [0.9019608 , 0.44705883, 0.13725491],
            11 : [0.05098039, 0.7372549 , 0.24313726],
            12 : [0.9019608 , 0.7137255 , 0.08627451]
        }
    else:
        cp = {
            0 : [255, 255, 255],
            1 : [0, 0, 0],
            2 : [0,0,17],
            3 : [0,0,34],
            4 : [0,0,51],
            5 : [0,0,68],
            6 : [0,0,85],
            7 : [0,0,102],
            8 : [0,0,119],
            9 : [0,0,136],
            10 : [0,0,153],
            11 : [0,0,170],
            12 : [0,0,187],
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

    if render_type == "normal":
        # depan
        # plt.subplot(2, n*2, (2*i)+1)
        ax.imshow(X_valid[vl_idx[i]][0].permute(1, 2, 0))
        ax.imshow(map_clr(y_predv[vl_idx[i]][0].argmax(axis=0).numpy()), alpha=0.5)
        plt.savefig('./static/img_front_result.png')

        # belakang
        # plt.subplot(2, n*2, (2*i)+2)
        ax.imshow(X_valid[vl_idx[i]][1].permute(1, 2, 0))
        ax.imshow(map_clr(y_predv[vl_idx[i]][1].argmax(axis=0).numpy()), alpha=0.5)
        plt.savefig('./static/img_back_result.png')
    else:
        # depan
        # plt.subplot(2, n*2, (2*i)+1)
        ax.imshow(map_clr(y_predv[vl_idx[i]][0].argmax(axis=0).numpy()), alpha=1)
        plt.savefig('./static/crop_img/crop_f.png')

        # belakang
        # plt.subplot(2, n*2, (2*i)+2)
        ax.imshow(map_clr(y_predv[vl_idx[i]][1].argmax(axis=0).numpy()), alpha=1)
        plt.savefig('./static/crop_img/crop_b.png')


def crop(img_direction):

    if img_direction == "front":
        img_path = "./static/crop_img/crop_f.png"
        save_path = "./static/crop_img/front/"
    else:
        img_path = "./static/crop_img/crop_b.png"
        save_path = "./static/crop_img/back/"

    img = Image.open(img_path)
    img = img.convert("RGBA")

    d = img.getdata()

    bone_name = ["Skull", "CervicalVert", "ThoracicVert", "Ribs", "Strenum", "Clavicle", "Scapula", "Humerus", "LumbarVert", "Sacrum", "Pelvis", "Femur"]
    color = [0, 17, 34, 51, 68, 85, 102, 119, 136, 153, 170, 187]

    index_bone_name = 0

    for c in color:
        new_image = []
        img = Image.open(img_path)
        img = img.convert("RGBA")
        for item in d:
            if item[2] in list(range(c,c+1)):
                if c == 0:
                    new_image.append((21, 193, 78, 190))
                elif c == 17:
                    new_image.append((0, 121, 255, 190))
                elif c == 34:
                    new_image.append((0, 228, 255, 190))
                elif c == 85:
                    new_image.append((109, 21, 193, 190))
                elif c == 51:
                    new_image.append((228, 70, 206, 190))
                elif c == 102:
                    new_image.append((224, 131, 46, 190))
                elif c == 68:
                    new_image.append((129, 64, 4, 190))
                elif c == 119:
                    new_image.append((14, 24, 156, 190))
                elif c == 136:
                    new_image.append((166, 5, 29, 190))
                elif c == 153:
                    new_image.append((255, 122, 0, 190))
                elif c == 170:
                    new_image.append((16, 119, 7, 190))
                elif c == 187:
                    new_image.append((225, 235, 52, 190))
                # new_image.append((55, 0, 75))
            else:
                new_image.append((255,255,255,0))

        # update image data
        img.putdata(new_image)

        # save new image
        # file_format = ".png"
        # img.save( save_path + bone_name[index_bone_name] + file_format)
        img.save(f"{save_path+bone_name[index_bone_name]}.png")
        index_bone_name += 1

def final_render():
    bone_name = ["Skull", "CervicalVert", "ThoracicVert", "Ribs", "Strenum", "Clavicle", "Scapula", "Humerus", "LumbarVert", "Sacrum", "Pelvis", "Femur"]

    # img = Image.open('upload/img/img_front.png')

    # # Pasting img2 image on top of img1 
    # # starting at coordinates (0, 0)
    # img.paste(img_decoded, (-35,8), mask=img_decoded)
    # img.save("static/final_front.png")

    # img = Image.open('upload/img/img_back.png')
    # # response = requests.get(decoded_back)
    # # img_decoded = Image.open('upload\img\img_back.png')
    # img_decoded = Image.open('image_back_done.png')

    # # Pasting img2 image on top of img1 
    # # starting at coordinates (0, 0)
    # img.paste(img_decoded, (-35,8), mask=img_decoded)
    # img.save("static/final_back.png")

    # make image list (back)
    img_list = []
    for b in bone_name:
        if b == "Skull":
            if exists(f'./static/img_annotation/back/{b}.png'):
                img = Image.open(f'./static/img_annotation/back/{b}.png')
            else:
                img = Image.open(f'./static/crop_img/back/{b}.png')
        else:
            if exists(f'./static/img_annotation/back/{b}.png'):
                img_list.append(Image.open(f'./static/img_annotation/back/{b}.png'))
            else:
                img_list.append(Image.open(f'./static/crop_img/back/{b}.png'))

    # merge img list (back)
    for i in range(len(img_list)):
        img.paste(img_list[i], (0,0), mask=img_list[i])

    img_raw = Image.open('upload/img/img_back.png').convert("RGBA")
    img_raw.paste(img, (-37,0), mask=img)
    img_raw.save("./static/final_back.png")

    # make image list (front)
    img_list = []
    for b in bone_name:
        if b == "Skull":
            if exists(f'./static/img_annotation/front/{b}.png'):
                img = Image.open(f'./static/img_annotation/front/{b}.png')
            else:
                img = Image.open(f'./static/crop_img/front/{b}.png')
        else:
            if exists(f'./static/img_annotation/front/{b}.png'):
                img_list.append(Image.open(f'./static/img_annotation/front/{b}.png'))
            else:
                img_list.append(Image.open(f'./static/crop_img/front/{b}.png'))

    # merge img list (front)
    for i in range(len(img_list)):
        img.paste(img_list[i], (0,0), mask=img_list[i])

    img_raw = Image.open('upload/img/img_front.png').convert("RGBA")
    img_raw.paste(img, (-37,0), mask=img)
    img_raw.save("./static/final_front.png")