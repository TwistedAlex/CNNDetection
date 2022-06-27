from networks.batch_GAIN_Deepfake import batch_GAIN_Deepfake
from networks.resnet import resnet50
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from torchvision.transforms import Normalize
from torchvision.transforms import Resize
from tqdm import tqdm
from util import save_roc_curve, save_roc_curve_with_threshold, show_cam_on_image, denorm
import argparse
import csv
import numpy as np
import os
import pathlib
import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import PIL.Image

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d','--dir', nargs='+', type=str, default='examples/realfakedir')
parser.add_argument('-n','--name', type=str, default='blur_jpg_prob0.5')
parser.add_argument('-m','--model_path', type=str, default='weights/blur_jpg_prob0.5.pth')
parser.add_argument('-b','--batch_size', type=int, default=32)
parser.add_argument('-j','--workers', type=int, default=4, help='number of workers')
parser.add_argument('-c','--crop', type=int, default=None, help='by default, do not crop. specify crop size')
parser.add_argument('--use_cpu', action='store_true', help='uses gpu by default, turn on to use cpu')
parser.add_argument('--size_only', action='store_true', help='only look at sizes of images in dataset')

opt = parser.parse_args()

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
roc_path = 'checkpoints/test/'+ opt.name + '/'
pathlib.Path(roc_path).mkdir(parents=True, exist_ok=True)

# Load model
if(not opt.size_only):
  model = resnet50(num_classes=1)
  if(opt.model_path is not None):
      state_dict = torch.load(opt.model_path, map_location='cpu')
  model.load_state_dict(state_dict['model'])
  model.eval()
  if(not opt.use_cpu):
      model.cuda()

  grad_layer = ["layer4"]
  num_classes = 1
  norm = Normalize(mean=mean, std=std)
  fill_color = norm(torch.tensor([0.4948, 0.3301, 0.16]).view(1, 3, 1, 1)).cuda()
  deepfake_model = batch_GAIN_Deepfake(model=model, grad_layer=grad_layer, num_classes=num_classes,
                              am_pretraining_epochs=10,
                              ex_pretraining_epochs=15,
                              fill_color=fill_color,
                              test_first_before_train=1,
                              grad_magnitude=1)
# Transform
trans_init = []
if(opt.crop is not None):
  trans_init = [transforms.CenterCrop(opt.crop),]
  print('Cropping to [%i]'%opt.crop)
else:
  print('Not cropping')
trans = transforms.Compose(trans_init + [
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset loader
if(type(opt.dir)==str):
  opt.dir = [opt.dir,]

print('Loading [%i] datasets'%len(opt.dir))
data_loaders = []
for dir in opt.dir:
  dataset = datasets.ImageFolder(dir, transform=trans)
  data_loaders+=[torch.utils.data.DataLoader(dataset,
                                          batch_size=opt.batch_size,
                                          shuffle=False,
                                          num_workers=opt.workers),]

y_true, y_pred = [], []
Hs, Ws = [], []

count = 0

with torch.no_grad():
  for data_loader in data_loaders:
    for data, label in tqdm(data_loader):
    # for data, label in data_loader:

      print(data.shape)
      print(label)
      Hs.append(data.shape[2])
      Ws.append(data.shape[3])

      y_true.extend(label.flatten().tolist())
      if(not opt.size_only):
        if(not opt.use_cpu):
            data = data.cuda()
        cur_y_pred = model(data).sigmoid().flatten().tolist()
        y_pred.extend(cur_y_pred)

        logits_cl, logits_am, heatmaps, masks, masked_images = deepfake_model(data, label)

        resize = Resize(size=224)
        for idx in range(data.shape[0]):
            htm = np.uint8(heatmaps[idx].squeeze().cpu().detach().numpy() * 255)
            orig = data[idx].permute([2, 0, 1])
            orig = resize(orig).permute([1, 2, 0])
            np_orig = orig.cpu().detach().numpy()
            orig = orig.unsqueeze(0)
            visualization, heatmap = show_cam_on_image(np_orig, htm, True)
            viz = torch.from_numpy(visualization).unsqueeze(0)
            masked_image = denorm(masked_images[idx].detach().squeeze(),
                                  mean, std)
            orig_viz = torch.cat((orig, viz, masked_image), 1)

            if label[idx] in [0]:
                PIL.Image.fromarray(orig_viz[0].cpu().numpy(), 'RGB').save(
                    roc_path + "/Neg/{:.7f}".format(cur_y_pred[idx].unsqueeze(0)[0][0]) + '_' + str(count) + '_gt_' + str(label[idx]) + '.png')
            if label[idx] in [1]:
                PIL.Image.fromarray(orig_viz[0].cpu().numpy(), 'RGB').save(
                    roc_path + "/Pos/{:.7f}".format(cur_y_pred[idx].unsqueeze(0)[0][0].cpu()) + '_' + str(count) + '_gt_' + str(label[idx]) + '.png')
            count += 1

Hs, Ws = np.array(Hs), np.array(Ws)
y_true, y_pred = np.array(y_true), np.array(y_pred)

save_roc_curve(y_true, y_pred, 0, roc_path)
save_roc_curve_with_threshold(y_true, y_pred, 0, roc_path)

print('Average sizes: [{:2.2f}+/-{:2.2f}] x [{:2.2f}+/-{:2.2f}] = [{:2.2f}+/-{:2.2f} Mpix]'.format(np.mean(Hs), np.std(Hs), np.mean(Ws), np.std(Ws), np.mean(Hs*Ws)/1e6, np.std(Hs*Ws)/1e6))
print('Num reals: {}, Num fakes: {}'.format(np.sum(1-y_true), np.sum(y_true)))

if(not opt.size_only):
  r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
  f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
  acc = accuracy_score(y_true, y_pred > 0.5)
  ap = average_precision_score(y_true, y_pred)

  print('AP: {:2.2f}, Acc: {:2.2f}, Acc (real): {:2.2f}, Acc (fake): {:2.2f}'.format(ap*100., acc*100., r_acc*100., f_acc*100.))


