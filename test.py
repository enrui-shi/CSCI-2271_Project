'''
load the trained model from save and using the test data to output result
'''
import numpy as np
import torch
import torch.nn as nn
import glob
import os
from PIL import Image
import rawpy
# from piq import ssim, SSIMLoss
import model
import unit


input_dir = './data/Sony/short/'
gt_dir = './data/Sony/long/'
result_dir = './result/9/'
model_PATH = './save/weights_2999.pth'

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# get test IDs
test_fns = glob.glob(gt_dir + '/1*.ARW')
test_ids = [int(os.path.basename(test_fn)[0:5]) for test_fn in test_fns]



model = model.Unet(4)
model.load_state_dict(torch.load(model_PATH))
model.eval()


for test_id in test_ids:
    # test the first image in each sequence
    in_files = glob.glob(input_dir + '%05d_00*.ARW' % test_id)
    print(in_files)
    for k in range(len(in_files)):
        in_path = in_files[k]
        in_fn = os.path.basename(in_path)
        gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % test_id)
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)

        raw = rawpy.imread(in_path)
        input_full = np.expand_dims(unit.pack_raw(raw), axis=0) * ratio

        im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        # scale_full = np.expand_dims(np.float32(im/65535.0),axis = 0)*ratio
        scale_full = np.expand_dims(np.float32(im / 65535.0), axis=0)
        gt_raw = rawpy.imread(gt_path)
        im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        gt_full = np.expand_dims(np.float32(im / 65535.0), axis=0)
        input_full = torch.from_numpy(np.minimum(input_full, 1.0))

        input_full = input_full.permute(0,3,1,2)


        output = model(input_full)
        output = output.permute(0,2,3,1)
        output = output.detach().numpy()
        output = np.minimum(np.maximum(output, 0), 1)
        

        output = output[0, :, :, :]
        gt_full = gt_full[0, :, :, :]
        scale_full = scale_full[0, :, :, :]
        scale_full = scale_full * np.mean(gt_full) / np.mean(scale_full)
        Image.fromarray((output*255).astype(np.uint8), mode = 'RGB').save(
            result_dir + 'final/%5d_00_%d_out.png' % (test_id, ratio))
        Image.fromarray((scale_full * 255).astype(np.uint8), mode = 'RGB').save(
            result_dir + 'final/%5d_00_%d_scale.png' % (test_id, ratio))
        Image.fromarray((gt_full * 255).astype(np.uint8), mode = 'RGB').save(
            result_dir + 'final/%5d_00_%d_gt.png' % (test_id, ratio))
        # scipy.misc.toimage(output * 255, high=255, low=0, cmin=0, cmax=255).save(
        #     result_dir + 'final/%5d_00_%d_out.png' % (test_id, ratio))
        # scipy.misc.toimage(scale_full * 255, high=255, low=0, cmin=0, cmax=255).save(
        #     result_dir + 'final/%5d_00_%d_scale.png' % (test_id, ratio))
        # scipy.misc.toimage(gt_full * 255, high=255, low=0, cmin=0, cmax=255).save(
        #     result_dir + 'final/%5d_00_%d_gt.png' % (test_id, ratio))