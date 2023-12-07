import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.io as io
import time
import sys
from PIL import Image

IS_COLAB = 'google.colab' in sys.modules

print(f"IS_COLAB: {IS_COLAB}")

OUTPUT_SHAPE = [512, 512]
PATCH_SHAPE = [16, 16]
BATCH_SIZE = 64
STACKING_SIZE = 2
LEARNING_RATE_D = 0.004
LEARNING_RATE_G = 0.001
SAVE_INTERVAL = 64
SRC_IMAGE = "grassflower.png"
PRINT_TIME = 5000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def img_int8tofloat(x):
    return x.float() / 255.0 * 2.0 - 1.0

if IS_COLAB:
    from google.colab import drive
    drive.mount('/content/gdrive')
    imgfilename = f"/content/gdrive/My Drive/texgen/input/{SRC_IMAGE}"
else:
    imgfilename = f"inputs/{SRC_IMAGE}"

real_img = Image.open(imgfilename)
real_img = transforms.ToTensor()(real_img).unsqueeze(0).to(device)
print(real_img.shape, real_img.dtype)

def realimg():
    ys = torch.randint(0, real_img.shape[2] - PATCH_SHAPE[0], (BATCH_SIZE*STACKING_SIZE,))
    xs = torch.randint(0, real_img.shape[3] - PATCH_SHAPE[1], (BATCH_SIZE*STACKING_SIZE,))
    
    out = []
    for bs_i in range(BATCH_SIZE*STACKING_SIZE):
        patch = real_img[:, :, ys[bs_i]:ys[bs_i] + PATCH_SHAPE[0], xs[bs_i]:xs[bs_i] + PATCH_SHAPE[1]]
        out.append(patch)
    
    ret = torch.cat(out, dim=0)
    ret = ret.view(BATCH_SIZE, STACKING_SIZE * 3, PATCH_SHAPE[0], PATCH_SHAPE[1])
    return ret

class FakeImg(nn.Module):
    def __init__(self):
        super(FakeImg, self).__init__()
        self.img = nn.Parameter(torch.zeros(1, 3, OUTPUT_SHAPE[0], OUTPUT_SHAPE[1]).to(device))

    def forward(self):
        processed_img = self.img
        processed_img = torch.cat([processed_img, processed_img[:, :, :PATCH_SHAPE[0] - 1, :]], dim=2)
        processed_img = torch.cat([processed_img, processed_img[:, :, :, :PATCH_SHAPE[1] - 1]], dim=3)
    
        ys = torch.randint(0, OUTPUT_SHAPE[0], (BATCH_SIZE * STACKING_SIZE,))
        xs = torch.randint(0, OUTPUT_SHAPE[1], (BATCH_SIZE * STACKING_SIZE,))
        
        out = []
        for bs_i in range(BATCH_SIZE*STACKING_SIZE):
            patch = processed_img[:, :, ys[bs_i]:ys[bs_i] + PATCH_SHAPE[0], xs[bs_i]:xs[bs_i] + PATCH_SHAPE[1]]
            out.append(patch)
        
        ret = torch.cat(out, dim=0).view(BATCH_SIZE, STACKING_SIZE * 3, PATCH_SHAPE[0], PATCH_SHAPE[1])
        return ret

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=3 * STACKING_SIZE, out_channels=24 * 4, kernel_size=3, padding='same'),
            nn.Conv2d(in_channels=24 * 4, out_channels=32 * 4, kernel_size=3, padding='same'),
            nn.Conv2d(in_channels=32 * 4, out_channels=64 * 4, kernel_size=3, padding='same')
        ])
        
        self.convs2 = nn.ModuleList([
            nn.Conv2d(in_channels=24 * 4, out_channels=24 * 4, kernel_size=3, padding='same'),
            nn.Conv2d(in_channels=32 * 4, out_channels=32 * 4, kernel_size=3, padding='same'),
            nn.Conv2d(in_channels=64 * 4, out_channels=64 * 4, kernel_size=3, padding='same')
        ])
        
        self.lns = nn.ModuleList([nn.LayerNorm([24 * 4, PATCH_SHAPE[0], PATCH_SHAPE[1]]),
                                  nn.LayerNorm([32 * 4, PATCH_SHAPE[0]//2, PATCH_SHAPE[1]//2]),
                                  nn.LayerNorm([64 * 4, PATCH_SHAPE[0]//4, PATCH_SHAPE[1]//4])])
        
        self.pools = nn.ModuleList([nn.AvgPool2d(kernel_size=2),
                                    nn.AvgPool2d(kernel_size=2),
                                    None])
        
        self.lastdense = nn.Linear(64 * 4 * (PATCH_SHAPE[0] // 4) * (PATCH_SHAPE[1] // 4), 1, bias=False)
                
    def forward(self, inputdata):
        for n in range(3):
            inputdata = torch.relu(self.convs[n](inputdata))
            inputdata = inputdata + torch.relu(self.convs2[n](self.lns[n](inputdata)))
            if self.pools[n] is not None:
                inputdata = self.pools[n](inputdata)
            
        inputdata = inputdata.view(inputdata.size(0), -1)
        inputdata = self.lastdense(inputdata)
        inputdata = inputdata.squeeze(1)
        return inputdata

fakeimg = FakeImg().to(device)
d = Discriminator().to(device)

optimizer_d = optim.Adam(d.parameters(), lr=LEARNING_RATE_D, amsgrad=True)
optimizer_g = optim.Adam(fakeimg.parameters(), lr=LEARNING_RATE_G, amsgrad=True)

iters = 0

def do_thing_D():
    with torch.no_grad():
        fi = fakeimg()
        ri = realimg()

    fakes = d(fi)
    reals = d(ri)
    
    reals = reals.unsqueeze(0)
    fakes = fakes.unsqueeze(1)
    return fakes - reals

def train_D():
    # train discriminator
    optimizer_d.zero_grad()
    loss = torch.nn.functional.softplus(do_thing_D())
    loss.mean().backward()
    optimizer_d.step()

def do_thing_G():
    with torch.no_grad():
        ri = realimg()

    fakes = d(fakeimg())
    reals = d(ri)
    
    reals = reals.unsqueeze(0)
    fakes = fakes.unsqueeze(1)
    return reals - fakes

def train_G():
    # train generator
    optimizer_g.zero_grad()
    loss = torch.nn.functional.relu(do_thing_G())
    loss.mean().backward()
    optimizer_g.step()

currtime = time.time()
curriters = 0
while True:
    iters += 1
    curriters += 1
    train_D()
    if iters >= 64:
        train_G()

    if (time.time() - currtime) * 1000.0 > PRINT_TIME:
        delta = time.time() - currtime
        
        print(f"#{iters}, {delta * 1000.0 / curriters} ms/iter")

        currtime = time.time()
        curriters = 0
    
    if iters % SAVE_INTERVAL == 0:
        img = (fakeimg.img.squeeze(0) + 1.0) * 127.5
        img = torch.clamp(img, 0.0, 255.0).to("cpu")
        img = img.byte()
        
        if IS_COLAB:
            io.write_png(img, f"/content/gdrive/My Drive/texgen/{iters}.png")
        else:
            io.write_png(img, f"outputs/{iters}.png")
