from btrfly import BtrflyNet
from PIL import Image
from torchvision import transforms

import matplotlib.pyplot as plt
import torch, os, timeit
import numpy as np

kupu = BtrflyNet()
kupu.load_state_dict( torch.load('./model-btr-paper.pt', map_location=torch.device('cpu')) )

tfms = transforms.Compose([transforms.ToTensor()])

imsize = (512, 128, 1)

def masking(msk):
    tmp = msk.round()
    return [(tmp == i).astype('float').tolist() for i in range(13)]

X, y = [], []

for name in os.listdir(f'./data/img'):
  if name[-5] == '1':
    X.append([
      tfms(Image.open(f'./data/img/{name[:-5]}1.png')).tolist(),
      tfms(Image.open(f'./data/img/{name[:-5]}2.png')).tolist()
    ])
    y.append([
      masking(plt.imread(f'./data/msk/{name[:-5]}1.png')[...,0] * 12),
      masking(plt.imread(f'./data/msk/{name[:-5]}2.png')[...,0] * 12)
    ])

vl = [ 6, 29,  0, 22, 34, 30,  4, 31]
tr = [i for i in range(37) if i not in vl]

X_train, X_valid = torch.Tensor( [X[i] for i in tr] ), torch.Tensor( [X[j] for j in vl] )
y_train, y_valid = torch.Tensor( [y[i] for i in tr] ), torch.Tensor( [y[j] for j in vl] )

n_data_train, n_data_valid = len(X_train), len(X_valid)

def masking_torch(msk):
    return torch.Tensor([ (msk == i).cpu().numpy().astype('float') for i in range(13) ])

inp_size = (1,  3, 512, 128)
out_size = (1, 13, 512, 128)
dsc_size = (1, 1, 13, 512, 128)

start = timeit.default_timer()

with torch.no_grad():

    for i in range(n_data_train):
        
        x, y = X_train[i], y_train[i]
        
        out_ant, out_pos = kupu(
            x[0].reshape(inp_size),
            x[1].reshape(inp_size)
        )

        tmp = torch.cat([
            masking_torch(out_ant.argmax(axis=1)[0]).reshape(dsc_size),
            masking_torch(out_pos.argmax(axis=1)[0]).reshape(dsc_size)
        ], axis=1)

        if i == 0: y_predt = tmp + 0
        else: y_predt = torch.cat([y_predt, tmp], axis=0)

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

stop = timeit.default_timer()

print('Time: ', stop - start)

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

def map_clr(mask):
    res = []
    for row in mask:
        new_row = [cp[x] for x in row]
        res.append(new_row)
    return np.array(res)

n = 4

np.random.seed(76)

tr_idx = np.random.choice(range(n_data_train), n, False)
vl_idx = np.random.choice(range(n_data_valid), n, False)

plt.figure(figsize=(15, n*3))
for i in range(n):
    
    plt.subplot(2, n*2, (2*i)+1)
    plt.imshow(X_train[tr_idx[i]][0].permute(1, 2, 0))
    plt.imshow(map_clr(y_predt[tr_idx[i]][0].argmax(axis=0).numpy()), alpha=0.5)

    plt.subplot(2, n*2, (2*i)+2)
    plt.imshow(X_train[tr_idx[i]][1].permute(1, 2, 0))
    plt.imshow(map_clr(y_predt[tr_idx[i]][1].argmax(axis=0).numpy()), alpha=0.5)

    plt.subplot(2, n*2, (n*2)+(2*i)+1)
    plt.imshow(X_train[tr_idx[i]][0].permute(1, 2, 0))
    plt.imshow(map_clr(y_train[tr_idx[i]][0].argmax(axis=0).numpy()), alpha=0.5)

    plt.subplot(2, n*2, (n*2)+(2*i)+2)
    plt.imshow(X_train[tr_idx[i]][1].permute(1, 2, 0))
    plt.imshow(map_clr(y_train[tr_idx[i]][1].argmax(axis=0).numpy()), alpha=0.5)

# plt.savefig('./btr_100.png')
plt.savefig('./static/btr_100.png')

plt.figure(figsize=(15, n*3))
for i in range(n):
    
    plt.subplot(2, n*2, (2*i)+1)
    plt.imshow(X_valid[vl_idx[i]][0].permute(1, 2, 0))
    plt.imshow(map_clr(y_predv[vl_idx[i]][0].argmax(axis=0).numpy()), alpha=0.5)

    plt.subplot(2, n*2, (2*i)+2)
    plt.imshow(X_valid[vl_idx[i]][1].permute(1, 2, 0))
    plt.imshow(map_clr(y_predv[vl_idx[i]][1].argmax(axis=0).numpy()), alpha=0.5)

    plt.subplot(2, n*2, (n*2)+(2*i)+1)
    plt.imshow(X_valid[vl_idx[i]][0].permute(1, 2, 0))
    plt.imshow(map_clr(y_valid[vl_idx[i]][0].argmax(axis=0).numpy()), alpha=0.5)

    plt.subplot(2, n*2, (n*2)+(2*i)+2)
    plt.imshow(X_valid[vl_idx[i]][1].permute(1, 2, 0))
    plt.imshow(map_clr(y_valid[vl_idx[i]][1].argmax(axis=0).numpy()), alpha=0.5)

plt.savefig('./static/btr_200.png')

def dice_coef(y_true, y_pred):
    y_true_f = y_true[:,:,1:,:,:].flatten()
    y_pred_f = y_pred[:,:,1:,:,:].flatten()
    union = y_true_f.flatten().sum().item() + y_pred_f.flatten().sum().item()
    if union == 0: return 1
    intersection = (y_true_f * y_pred_f).sum().item()
    return 2 * intersection / union

dice_coef(y_train, y_predt)

dice_coef(y_valid, y_predv)

def dice_coef_all(y_true, y_pred):
    dice_scores = []
    for i in range(1,13):
        y_true_f = y_true[:,:,i,:,:].flatten()
        y_pred_f = y_pred[:,:,i,:,:].flatten()
        union = y_true_f.flatten().sum().item() + y_pred_f.flatten().sum().item()
        if union == 0:
            dice_scores.append(1)
        else:
            intersection = (y_true_f * y_pred_f).sum().item()
            dice_scores.append(2 * intersection / union)
    return dice_scores

import pandas as pd

segments = [
    'Skull (Green)',
    'Cervical Vertebrae (Blue)',
    'Thoracic Vertebrae (Light Blue)',
    'Ribs (Purple)',
    'Sternum (Pink)',
    'Collarbones (Light Brown)',
    'Shoulder Blades (Dark Brown)',
    'Humerus (Dark Blue)',
    'Lumbar Vertebrae (Yellow)',
    'Sacrum (Orange)',
    'Pelvis (Dark Green)',
    'Femur (Gold)'
]

pd.DataFrame({
    'Segment'    : segments,
    'Dice Score' : dice_coef_all(y_valid, y_predv)
})

def dice_coef_section(y_true, y_pred, sec):
    sec = {'ant' : 0, 'pos' : 1}[sec]
    dice_scores = []
    for i in range(1,13):
        y_true_f = y_true[:,sec,i,:,:].flatten()
        y_pred_f = y_pred[:,sec,i,:,:].flatten()
        union = y_true_f.flatten().sum().item() + y_pred_f.flatten().sum().item()
        if union == 0:
            dice_scores.append(1)
        else:
            intersection = (y_true_f * y_pred_f).sum().item()
            dice_scores.append(2 * intersection / union)
    return dice_scores

# anterior
pd.DataFrame({
    'Segment'    : segments,
    'Dice Score' : dice_coef_section(y_valid, y_predv, 'ant')
})

# posterior
pd.DataFrame({
    'Segment'    : segments,
    'Dice Score' : dice_coef_section(y_valid, y_predv, 'pos')
})