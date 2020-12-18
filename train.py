'''
Train for the model.
'''
import numpy as np
import torch
import torch.nn as nn
import glob
import os
import rawpy
import model
import unit



def train_model(model,optimizer,L,log_file, num_epochs=8000, ps = 512):
    a = True
    for epoch in range(num_epochs):
        # after traing 2000 epoch chenge lr to 1e-5
        if a and epoch > 2000:
            for g in optimizer.param_groups:
                g['lr'] = 0.00001
            a = False
        epoch_loss = 0
        step = 0
        dt_size = len(train_ids)
        for ind in np.random.permutation(len(train_ids)):
            step  += 1
            input_patch, gt = loaddata(ind,ps)
            optimizer.zero_grad()
            output = model(input_patch)
            loss = L(output,gt)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # print("%d/%d,train_loss:%0.3f" % (step, len(train_ids) + 1, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss/step))
        log_file.write("epoch %d loss:%0.3f\n" % (epoch, epoch_loss/step))
        if (epoch+1) % 200 == 0:
            log_file.flush()
            torch.save(model.state_dict(), './save/weights_%d.pth' % epoch) 
    return model



# load data
def loaddata(ind,ps):
    train_id = train_ids[ind]
    in_files = glob.glob(input_dir + '%05d_00*.ARW' % train_id)
    in_path = in_files[np.random.randint(0, len(in_files))]
    in_fn = os.path.basename(in_path)
    gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % train_id)
    gt_path = gt_files[0]
    gt_fn = os.path.basename(gt_path)
    in_exposure = float(in_fn[9:-5])
    gt_exposure = float(gt_fn[9:-5])
    ratio = min(gt_exposure / in_exposure, 300)
    if input_images[str(ratio)[0:3]][ind] is None:
        raw = rawpy.imread(in_path)
        input_patch = np.expand_dims(unit.pack_raw(raw), axis=0) * ratio
        gt_raw = rawpy.imread(gt_path)
        im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        gt_patch = np.expand_dims(np.float32(im / 65535.0), axis=0)
        input_images[str(ratio)[0:3]][ind] = input_patch
        gt_images[ind] = gt_patch
    else:
        input_patch = input_images[str(ratio)[0:3]][ind]
        gt_patch = gt_images[ind]

    
    H = input_patch.shape[1]
    W = input_patch.shape[2]


    xx = np.random.randint(0, W - ps)
    yy = np.random.randint(0, H - ps)
    input_patch = input_patch[:, yy:yy + ps, xx:xx + ps, :]
    gt_patch = gt_patch[:, yy * 2:yy * 2 + ps * 2, xx * 2:xx * 2 + ps * 2, :]

    if np.random.randint(2, size=1)[0] == 1:  # random flip
        input_patch = np.flip(input_patch, axis=1)
        gt_patch = np.flip(gt_patch, axis=1)
    if np.random.randint(2, size=1)[0] == 1:
        input_patch = np.flip(input_patch, axis=2)
        gt_patch = np.flip(gt_patch, axis=2)
    if np.random.randint(2, size=1)[0] == 1:  # random transpose
        input_patch = np.transpose(input_patch,(0, 2, 1, 3))
        gt_patch = np.transpose(gt_patch,(0, 2, 1, 3))
    input_patch = np.minimum(input_patch, 1.0)
    input_patch = torch.from_numpy(input_patch.copy()).to(device)
    gt_patch = torch.from_numpy(gt_patch.copy()).to(device)
    # input_patch = torch.minimum(input_patch,torch.ones(input_patch.shape).to(device))

    input_patch = input_patch.permute(0,3,1,2)
    gt_patch = gt_patch.permute(0,3,1,2)
    return input_patch, gt_patch







device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
# assign memory for raw data to store
gt_images = [None] * 6000
input_images = {}
input_images['300'] = [None] * len(train_ids)
input_images['250'] = [None] * len(train_ids)
input_images['100'] = [None] * len(train_ids)



# traing
f = open("log_path", "w")
learning_rate = 0.0001
model = model.Unet(4).to(device)
model.train()
L = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
train_model(model,optimizer,L,f)




