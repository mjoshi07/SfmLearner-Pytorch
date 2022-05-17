import os
import torch

from imageio import imread, imsave
from skimage.transform import resize as imresize
from skimage.transform import resize
from scipy.ndimage import zoom
import numpy as np
from path import Path
import argparse
from tqdm import tqdm
from imageio import imread
import matplotlib.pyplot as plt

from models import DispNetS, DispNetS2, PoseExpNet
from utils import tensor2array

EPSILON = 1e-6

parser = argparse.ArgumentParser(description='Script for DispNet testing with corresponding groundTruth',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dispnet1", required=True, type=str, help="pretrained Original DispNet path")
parser.add_argument("--dispnet2", required=True, type=str, help="pretrained Modified DispNet path")
parser.add_argument("--pretrained-posenet", default=None, type=str, help="pretrained PoseNet path (for scale factor)")
parser.add_argument("--img-height", default=128, type=int, help="Image height")
parser.add_argument("--img-width", default=416, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
parser.add_argument("--min-depth", default=1e-3)
parser.add_argument("--max-depth", default=80)

parser.add_argument("--data", default='.', type=str, help="Dataset directory")


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def main():
    args = parser.parse_args()

    disp_net1 = DispNetS().to(device)
    weights1 = torch.load(args.dispnet1,map_location='cpu')
    disp_net1.load_state_dict(weights1['state_dict'])
    disp_net1.eval()

    disp_net2 = DispNetS2().to(device)
    weights2 = torch.load(args.dispnet2,map_location='cpu')
    disp_net2.load_state_dict(weights2['state_dict'])
    disp_net2.eval()

    if args.pretrained_posenet is None:
        print('no PoseNet specified, scale_factor will be determined by median ratio, which is kiiinda cheating\
            (but consistent with original paper)')
        seq_length = 1
    else:
        weights = torch.load(args.pretrained_posenet)
        seq_length = int(weights['state_dict']['conv1.0.weight'].size(1)/3)
        pose_net = PoseExpNet(nb_ref_imgs=seq_length - 1, output_exp=False).to(device)
        pose_net.load_state_dict(weights['state_dict'], strict=False)

    dataset_dir = Path(args.data)

    output_dir = os.path.join(dataset_dir, "OUTPUT2")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    rgb_images = os.path.join(dataset_dir, "TEST")
    gt_images = os.path.join(dataset_dir, "GT")

    errors = np.zeros((2, 9, len(os.listdir(rgb_images))), np.float32)  

    for j, file in enumerate(tqdm(os.listdir(rgb_images))):
        
        basename = file.split('leftImg8bit')[0]
        gt_filename =  basename + 'disparity.png' 
        input_img = imread(os.path.join(rgb_images, file))
        gt = imread(os.path.join(gt_images, gt_filename))
        gt = resize(gt, (args.img_height, args.img_width))
        gt = gt.astype(np.float32)

        h,w,_ = input_img.shape
        if (not args.no_resize) and (h != args.img_height or w != args.img_width):
            input_img = resize(input_img, (args.img_height, args.img_width))
        img = np.transpose(input_img, (2, 0, 1))

        tensor_img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
        tensor_img = ((tensor_img - 0.5)/0.5).to(device)

        pred_disp1 = disp_net1(tensor_img)[0]
        disp1 = (255*tensor2array(pred_disp1, max_value=None, colormap='bone')).astype(np.uint8)
        disp1 = np.transpose(disp1, (1,2,0))[:, :, 0]
        pred_depth1 = 1/(disp1 + EPSILON)

        pred_disp2 = disp_net2(tensor_img)[0]
        disp2 = (255*tensor2array(pred_disp2, max_value=None, colormap='bone')).astype(np.uint8)
        disp2 = np.transpose(disp2, (1,2,0))[:, :, 0]
        pred_depth2 = 1/(disp2 + EPSILON)

        gt_depth = (0.209313 * 2262.52)/(gt + EPSILON)
        disp1 = (0.209313 * 2262.52)/(disp1 + EPSILON)
        disp2 = (0.209313 * 2262.52)/(disp2 + EPSILON)

        fig = plt.figure()

        plt.xlim((0, 100))
        plt.axis('off')
        fig.add_subplot(4, 1, 1)
        plt.imshow(input_img)
        plt.axis("off")   # turns off axes
        plt.axis("tight")  # gets rid of white border
        plt.axis("image")  # square up the image instead of filling the "figure" space
        fig.tight_layout()

        fig.add_subplot(4, 1, 2)
        plt.imshow(gt_depth, cmap='gray')
        plt.axis("off")   # turns off axes
        plt.axis("tight")  # gets rid of white border
        plt.axis("image")  # square up the image instead of filling the "figure" space
        fig.tight_layout()

        fig.add_subplot(4, 1, 3)
        plt.imshow(disp1, cmap='gray')
        plt.axis("off")   # turns off axes
        plt.axis("tight")  # gets rid of white border
        plt.axis("image")  # square up the image instead of filling the "figure" space
        fig.tight_layout()

        fig.add_subplot(4, 1, 4)
        plt.imshow(disp2, cmap='gray')
        plt.axis("off")   # turns off axes
        plt.axis("tight")  # gets rid of white border
        plt.axis("image")  # square up the image instead of filling the "figure" space
        fig.tight_layout()
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.show()
        # plt.savefig(os.path.join(output_dir, basename + "output_stacked.png"), dpi=500,bbox_inches='tight',pad_inches = 0)
        plt.clf()

        # scale_factor1 = np.median(gt_depth)/np.median(pred_depth1)
        # errors[0,:,j] = compute_errors(1 / scale_factor1 *gt_depth, scale_factor1 * pred_depth1)

        # scale_factor2 = np.median(gt_depth)/np.median(pred_depth2)
        # errors[1,:,j] = compute_errors(1 / scale_factor2 * gt_depth, scale_factor2 * pred_depth2)

        scale_factor1 = np.median(gt_depth)/np.median(disp1)
        errors[0,:,j] = compute_errors(disp1, disp2)

        # scale_factor2 = np.median(gt_depth)/np.median(disp2)
        # errors[1,:,j] = compute_errors(gt_depth, scale_factor2*disp2)

    mean_errors = errors.mean(2)
    print(mean_errors[0])
    # print(mean_errors[1][:4])

 
def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_log = np.mean(np.abs(np.log(gt) - np.log(pred)))

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    abs_diff = np.mean(np.abs(gt - pred))

    sq_rel = np.mean(((gt - pred)**2) / gt)

    return abs_diff, abs_rel, sq_rel, rmse, rmse_log, abs_log, a1, a2, a3


if __name__ == '__main__':
    main()
