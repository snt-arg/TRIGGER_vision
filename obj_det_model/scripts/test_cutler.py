import numpy as np
import PIL.Image as Image
import torch
from torchvision import transforms
from scipy import ndimage
import matplotlib.pyplot as plt

from CutLER.third_party.TokenCut.unsupervised_saliency_detection import metric
from CutLER.maskcut import dino
from CutLER.maskcut.maskcut import maskcut
from CutLER.maskcut.colormap import random_color
from CutLER.maskcut.crf import densecrf


def vis_mask(input, mask, mask_color) :
    fg = mask > 0.5
    rgb = np.copy(input)
    rgb[fg] = (rgb[fg] * 0.3 + np.array(mask_color) * 0.7).astype(np.uint8)
    return Image.fromarray(rgb)

def main():
    # Image transformation applied to all images
    ToTensor = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(
                                    (0.485, 0.456, 0.406),
                                    (0.229, 0.224, 0.225)),])


    # DINO hyperparameters
    vit_arch = 'base'
    vit_feat = 'k'
    patch_size = 8
    # DINO pre-trained model
    url = "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
    feat_dim = 768
    # MaskCut hyperparameters
    fixed_size = 480
    tau = 0.15
    N = 3
    # use cpu (cpu=True) or gpu (cpu=False)
    cpu = False
    # demo image path; you can change the img_path to try other demos or your own images.
    img_path = './taco_dataset/data/batch_1/000021.jpg'
    # img_path = './CutLER/maskcut/imgs/demo1.jpg'
    # extract patch features with a pretrained DINO model
    backbone = dino.ViTFeat(url, feat_dim, vit_arch, vit_feat, patch_size)
    msg = 'Load {} pre-trained feature...'.format(vit_arch)
    backbone.eval()
    if not cpu:
        backbone.cuda()

    # get pesudo-masks with MaskCut
    bipartitions, _, I_new = maskcut(img_path, backbone, patch_size, tau, N=N, fixed_size=fixed_size, cpu=cpu)
    I = Image.open(img_path).convert('RGB')
    width, height = I.size
    pseudo_mask_list = []
    for idx, bipartition in enumerate(bipartitions):
        # post-process pesudo-masks with CRF
        pseudo_mask = densecrf(np.array(I_new), bipartition)
        pseudo_mask = ndimage.binary_fill_holes(pseudo_mask>=0.5)

        # filter out the mask that have a very different pseudo-mask after the CRF
        if not cpu:
            mask1 = torch.from_numpy(bipartition).cuda()
            mask2 = torch.from_numpy(pseudo_mask).cuda()
        else:
            mask1 = torch.from_numpy(bipartition)
            mask2 = torch.from_numpy(pseudo_mask)
        if metric.IoU(mask1, mask2) < 0.5:
            pseudo_mask = pseudo_mask * -1

        # construct binary pseudo-masks
        pseudo_mask[pseudo_mask < 0] = 0
        pseudo_mask = Image.fromarray(np.uint8(pseudo_mask*255))
        pseudo_mask = np.asarray(pseudo_mask.resize((width, height)))

        pseudo_mask = pseudo_mask.astype(np.uint8)
        upper = np.max(pseudo_mask)
        lower = np.min(pseudo_mask)
        thresh = upper / 2.0
        pseudo_mask[pseudo_mask > thresh] = upper
        pseudo_mask[pseudo_mask <= thresh] = lower
        pseudo_mask_list.append(pseudo_mask)

    # pseudo-mask visualization
    # pseudo-mask visualization
    image = np.array(I)
    for pseudo_mask in pseudo_mask_list:
        image = vis_mask(image, pseudo_mask, random_color(rgb=True))
    plt.imshow(image)
    plt.show()



if __name__ == "__main__":
    main()
