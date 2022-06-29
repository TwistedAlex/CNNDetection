from networks.batch_GAIN_Deepfake import batch_GAIN_Deepfake
from networks.resnet import resnet50
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from torchvision.transforms import Normalize
from torchvision.transforms import Resize
from tqdm import tqdm
from util import show_cam_on_image, denorm
import PIL.Image
import argparse
import csv
import numpy as np
import os
import pathlib
import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

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
device = torch.device('cuda:'+str(0))
roc_path = 'checkpoints/test/'+ opt.name + '/'
pathlib.Path(roc_path+'/Neg/').mkdir(parents=True, exist_ok=True)
pathlib.Path(roc_path+'/Pos/').mkdir(parents=True, exist_ok=True)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
norm = Normalize(mean=mean, std=std)
resize = Resize(size=224)
# Load model
if(not opt.size_only):
    model = resnet50(num_classes=1)

    fill_color = norm(torch.tensor([0.4948,0.3301,0.16]).view(1, 3, 1, 1)).cuda()
    grad_layer = ["layer4"]
    if(opt.model_path is not None):
        state_dict = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model = batch_GAIN_Deepfake(model=model, grad_layer=grad_layer, num_classes=1,
                              am_pretraining_epochs=1,
                              ex_pretraining_epochs=1,
                              fill_color=fill_color,
                              test_first_before_train=0,
                              grad_magnitude=1)
    if(not opt.use_cpu):
        model.cuda()
model.eval()
# Transform
trans_init = []
if(opt.crop is not None):
    # trans_init = [transforms.CenterCrop(opt.crop),]
    print('Cropping to [%i]'%opt.crop)
else:
    print('Not cropping')
trans = transforms.Compose(trans_init + [
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset loader
if(type(opt.dir)==str):
    opt.dir = [opt.dir,]

print('Loading [%i] datasets'%len(opt.dir))
data_loaders = []
for dir in opt.dir:
    dataset = datasets.ImageFolder(dir, transform=trans)
    print(type(dataset))
    print(len(dataset))
    print(len(dataset[0]))
    print(dataset[0][0])
    print(dataset[0][0].shape)
    print(dataset[0][1])
    data_loaders+=[torch.utils.data.DataLoader(dataset,
                                          batch_size=opt.batch_size,
                                          shuffle=False,
                                          num_workers=opt.workers),]

    PIL.Image.fromarray(dataset[0][0].cpu().numpy(), 'RGB').save(
        roc_path + "/firstEle_dataset.png")
y_true, y_pred = [], []
Hs, Ws = [], []
count = 0

for data_loader in data_loaders:
    for data, label in tqdm(data_loader):
    # for data, label in data_loader:
        Hs.append(data.shape[2])
        Ws.append(data.shape[3])

        cur_y_true = label.flatten().tolist()
        labels = torch.Tensor(cur_y_true).to(device).float()
        y_true.extend(cur_y_true)
        if(not opt.size_only):
            if(not opt.use_cpu):
                data = data.cuda()

            logits_cl, logits_am, heatmaps, masks, masked_images =  model(data, labels)
            print("debug**********")
            print(data)  # [[ [[],[]..[]], [[],[]...[]] ]]
            print(labels) # tensor([1], device='cuda:0')
            print(data.shape) # torch.size([1, 3, 224, 224])
            print(labels.shape) # torch.size([1])
            print(logits_cl.shape) # [1, 2]
            print(logits_am.shape) # [1, 2]
            print(heatmaps.shape) # [1, 1, 224, 224]
            print(masks.shape) # [1, 1, 224, 224]
            print(masked_images.shape) # [1, 3, 224, 224]
            y_pred.extend(logits_cl.sigmoid().flatten().tolist())

            for idx in (range(opt.batch_size)):
                htm = np.uint8(heatmaps[idx][0].squeeze().cpu().detach().numpy() * 255)
                orig = data[idx] # data[idx] target [1024, 1024, 3]
                orig = orig.permute([1, 2, 0])
                np_orig = orig.cpu().detach().numpy()
                print("np_orig, htm")
                print(np_orig.shape) # 224,224,3 now 224, 16725, 224
                print(htm.shape) # 224, 224 now 224, 224
                visualization, heatmap = show_cam_on_image(np_orig, htm, True)
                print("visualization, heatmap")
                print(visualization.shape)
                print(heatmap.shape)
                viz = torch.from_numpy(visualization).unsqueeze(0).to(device)
                orig = orig.unsqueeze(0)
                print("viz, orig")
                print(viz.shape) # [224,224,3]
                print(orig.shape)
                orig_viz = torch.cat((orig, viz), 1)

                if label[idx] == 0:
                    print('0')
                    PIL.Image.fromarray(orig_viz[0].cpu().numpy(), 'RGB').save(
                        roc_path + "/Neg/{:.7f}".format(y_pred[count]) + '_' + str(count) + '_gt_' + str(y_true[count]) + '.png')
                if label[idx] == 1:
                    print('1')
                    PIL.Image.fromarray(orig_viz[0].cpu().numpy(), 'RGB').save(
                        roc_path + "/Pos/{:.7f}".format(y_pred[count]) + '_' + str(count) + '_gt_' + str(y_true[count]) + '.png')
                count += 1
                exit(0)

Hs, Ws = np.array(Hs), np.array(Ws)
y_true, y_pred = np.array(y_true), np.array(y_pred)

print('Average sizes: [{:2.2f}+/-{:2.2f}] x [{:2.2f}+/-{:2.2f}] = [{:2.2f}+/-{:2.2f} Mpix]'.format(np.mean(Hs), np.std(Hs), np.mean(Ws), np.std(Ws), np.mean(Hs*Ws)/1e6, np.std(Hs*Ws)/1e6))
print('Num reals: {}, Num fakes: {}'.format(np.sum(1-y_true), np.sum(y_true)))

if(not opt.size_only):
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)

    print('AP: {:2.2f}, Acc: {:2.2f}, Acc (real): {:2.2f}, Acc (fake): {:2.2f}'.format(ap*100., acc*100., r_acc*100., f_acc*100.))

