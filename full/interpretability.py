import os
import torch
import torch.utils.data
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
import glob
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import model
from util.preprocess import mean, std, preprocess_input_function
from util.deletion_auc import CausalMetric, auc
from util.iou import iou_metric


parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0')
args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print("GPU ID:", os.environ['CUDA_VISIBLE_DEVICES'])


from settings_CUB_DOG import img_size, prototype_shape, num_classes, prototype_activation_function, add_on_layers_type


base_architecture = 'vgg19'

# construct the model
ppnet = model.construct_STProtoPNet(base_architecture=base_architecture,
                                    pretrained=True, img_size=img_size,
                                    prototype_shape=prototype_shape,
                                    num_classes=num_classes,
                                    prototype_activation_function=prototype_activation_function,
                                    add_on_layers_type=add_on_layers_type)
ppnet = ppnet.cuda()
class_specific = True


# pretrained model
checkpoint_path = "80nopush0.8024_vgg19_40p.pth"
ppnet.load_state_dict(torch.load(checkpoint_path))

model = torch.nn.DataParallel(ppnet)
model.eval()


# all datasets
num_workers = 0  # 20, 8
normalize = transforms.Normalize(mean=mean, std=std)
transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
            ])

test_batch_size = 1
test_dir = '/mnt/c/chong/data/CUB_200_2011_full/test/'
test_dataset = datasets.ImageFolder(
    test_dir,
    transform,
    )
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=False)


GT_PATH = '/mnt/c/chong/data/CUB_200_2011_full/mask_test_binary/'

deletion = CausalMetric(model, 'del', step=224 * 8, substrate_fn=torch.zeros_like)

del_all = []
ch_all = []
iou_all = []
oirr_all = []
for i, (img_name, label) in enumerate(test_dataset.imgs):

    img_ori_PIL = Image.open(img_name)
    img_ori_PIL = img_ori_PIL.convert('RGB')
    img = transform(img_ori_PIL)

    gt_seg_path = os.path.join(GT_PATH, '/'.join(img_name.split('/')[-2:]).replace('.jpg', '.png'))
    gt_mask = cv2.imread(gt_seg_path)[:, :, 0]
    gt_mask[gt_mask != 0] = 1

    img_np = img.cpu().numpy().transpose(1, 2, 0).astype(np.float32)
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

    input = img.unsqueeze(0).cuda()
    target = torch.tensor(label).cuda()
    with torch.no_grad():
        _, _, similarity_maps = model(input)

    num_proto_per_class = model.module.num_prototypes_per_class
    prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:, label]).cuda()
    proto_act_img_0 = similarity_maps[0][:, prototypes_of_correct_class == 1, ...].squeeze().detach().cpu().numpy()
    proto_act_img_1 = similarity_maps[1][:, prototypes_of_correct_class == 1, ...].squeeze().detach().cpu().numpy()

    proto_act_img_0 = proto_act_img_0.mean(0)
    proto_act_img_1 = proto_act_img_1.mean(0)

    comb_w = 0.5
    proto_act = (comb_w * (proto_act_img_0) + (1 - comb_w) * (proto_act_img_1))


    heatmap_deletion = cv2.resize(proto_act, dsize=(img_size, img_size), interpolation=cv2.INTER_CUBIC)  # model input resolution
    heatmap_deletion = (heatmap_deletion - heatmap_deletion.min()) / (heatmap_deletion.max() - heatmap_deletion.min())

    del_score = deletion.single_run(input, heatmap_deletion, verbose=0)   # verbose=2
    del_auc = auc(del_score)
    del_all.append(del_auc)

    heatmap = cv2.resize(proto_act, dsize=(img_ori_PIL.size[0], img_ori_PIL.size[1]), interpolation=cv2.INTER_CUBIC)  # original image resolution
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    ch = heatmap[gt_mask == 1].sum() / (gt_mask == 1).sum()
    oirr = (heatmap[gt_mask == 0].sum() / (gt_mask == 0).sum()) / (heatmap[gt_mask == 1].sum() / (gt_mask == 1).sum())
    iou = iou_metric(heatmap, gt_mask)

    ch_all.append(ch)
    oirr_all.append(oirr)
    iou_all.append(iou)

    if i % 100 == 0:
        print(i, len(test_dataset.imgs),
              "DAUC:", np.round(sum(del_all)/len(del_all), 4),
              "CH:", np.round(sum(ch_all)/len(ch_all), 4),
              "OIRR:", np.round(sum(oirr_all)/len(oirr_all), 4),
              "IoU:", np.round(sum(iou_all)/len(iou_all), 4),
              )


print('Number of samples:', len(del_all))
print('')
print('Mean DAUC:', np.round(sum(del_all)/len(del_all), 4))
print('Mean CH:', np.round(sum(ch_all)/len(ch_all), 4))
print('Mean OIRR:', np.round(sum(oirr_all)/len(oirr_all), 4))
print('Mean IoU:', np.round(sum(iou_all)/len(iou_all), 4))
