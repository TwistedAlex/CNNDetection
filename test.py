from networks.batch_GAIN_Deepfake import batch_GAIN_Deepfake
from networks.resnet import resnet50
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from torchvision.transforms import Normalize
from torchvision.transforms import Resize, RandomResizedCrop
from torchvision.transforms import ToTensor
from tqdm import tqdm
from util import save_roc_curve, save_roc_curve_with_threshold
from util import show_cam_on_image, denorm, select_clo_far_heatmaps
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


def test_output_heatmap(model, dir):
    pass


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d','--dir', nargs='+', type=str, default='examples/realfakedir')
parser.add_argument('-n','--name', type=str, default='blur_jpg_prob0.5')
parser.add_argument('-m','--model_path', type=str, default='weights/blur_jpg_prob0.5.pth')
parser.add_argument('-b','--batch_size', type=int, default=32)
parser.add_argument('-j','--workers', type=int, default=4, help='number of workers')
parser.add_argument('-c','--crop', type=int, default=None, help='by default, do not crop. specify crop size')
parser.add_argument('--default_test', action='store_true', help='default mode')
parser.add_argument('--use_cpu', action='store_true', help='uses gpu by default, turn on to use cpu')
parser.add_argument('--size_only', action='store_true', help='only look at sizes of images in dataset')

if __name__ == '__main__':
    opt = parser.parse_args()
    device = torch.device('cuda:'+str(0))
    heatmap_home_dir = "/server_data/image-research/"
    roc_path = 'checkpoints/test/'+ opt.name + '/'
    psi_05_test = "dataset/test_psi0.5_ffhq/"
    psi_1_test = "dataset/test_psi1_ffhq/"
    psi_05_input_path_heatmap = roc_path + '/test_heatmap/psi0.5/'
    psi_1_input_path_heatmap = roc_path + '/test_heatmap/psi1/'

    # pathlib.Path(roc_path+'/Neg/').mkdir(parents=True, exist_ok=True)
    # pathlib.Path(roc_path+'/Pos/').mkdir(parents=True, exist_ok=True)

    pathlib.Path(psi_05_input_path_heatmap+'/Neg/').mkdir(parents=True, exist_ok=True)
    pathlib.Path(psi_05_input_path_heatmap+'/Pos/').mkdir(parents=True, exist_ok=True)
    pathlib.Path(psi_1_input_path_heatmap+'/Neg/').mkdir(parents=True, exist_ok=True)
    pathlib.Path(psi_1_input_path_heatmap+'/Pos/').mkdir(parents=True, exist_ok=True)
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
        trans_init = [transforms.CenterCrop(opt.crop),]
        print('Cropping to [%i]'%opt.crop)
    else:
        print('Not cropping')
    trans = transforms.Compose(trans_init + [
        # RandomResizedCrop(224, scale=(0.88, 1.0), ratio=(0.999, 1.001)),
        transforms.ToTensor(), # (H x W x C) to (C x H x W), [0,255] to [0.0, 1.0]torch.FloatTensor # only-1024
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Dataset loader
    if(type(opt.dir)==str):
        dirs = [opt.dir,]
    if not opt.default_test:
        default_flag = True
    data_loaders = []
    if default_flag:
        dirs = [psi_05_test, psi_1_test]
    print('Loading [%i] datasets' % len(dirs))
    for dir in dirs:
        if 'psi1' in dir:
            mode = 'psi_1'
            htm_path = psi_1_input_path_heatmap
        else:
            mode = 'psi_0.5'
            htm_path = psi_05_input_path_heatmap
        print(f'Test path: {dir}')
        dataset = datasets.ImageFolder(dir, transform=trans)
        # print(type(dataset))
        # print(len(dataset))
        # print(len(dataset[0]))
        # print(dataset[0][0])
        # print(dataset[0][0].shape)
        # print(dataset[0][1])
        data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=False,
                                              num_workers=opt.workers)

        # PIL.Image.fromarray((dataset[1][0].permute([1, 2, 0]).cpu().numpy() * 255).astype('uint8'), 'RGB').save(
        #     roc_path + "/Neg/firstEle_dataset.png")
        y_true, y_pred = [], []
        Hs, Ws = [], []
        count = 0

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
                # print("debug**********")
                # print(data)  # [[ [[],[]..[]], [[],[]...[]] ]]
                # print(labels) # tensor([1], device='cuda:0')
                # print(data.shape) # torch.size([1, 3, 224, 224])
                # print(labels.shape) # torch.size([1])
                # print(logits_cl.shape) # [1, 2]
                # print(logits_am.shape) # [1, 2]
                # print(heatmaps.shape) # [1, 1, 224, 224]
                # print(masks.shape) # [1, 1, 224, 224]
                # print(masked_images.shape) # [1, 3, 224, 224]
                y_pred.extend(logits_cl.sigmoid().flatten().tolist())

                for idx in (range(opt.batch_size)):
                    htm = np.uint8(heatmaps[idx][0].squeeze().cpu().detach().numpy() * 255)
                    orig = data[idx] # data[idx] target [1024, 1024, 3]
                    orig = orig.permute([1, 2, 0])
                    np_orig = np.uint8(orig.cpu().detach().numpy() * 255)
                    PIL.Image.fromarray(np_orig, 'RGB').save(
                        roc_path + "/firstEle_dataset.png")
                    # print("np_orig, htm")
                    print(np_orig.shape) # 224,224,3 now 224, 16725, 224
                    print(htm.shape) # 224, 224 now 224, 224

                    visualization, heatmap = show_cam_on_image(np_orig, htm, True)
                    # print("visualization, heatmap")
                    # print(visualization.shape)
                    # print(heatmap.shape)
                    viz = torch.from_numpy(visualization).unsqueeze(0).to(device)
                    # PIL.Image.fromarray(viz[0].cpu().numpy(), 'RGB').save(
                    #     roc_path + "/Neg/viz0_totensor.png")
                    # PIL.Image.fromarray((viz[0].cpu().numpy() * 255).astype('uint8'), 'RGB').save(
                    #     roc_path + "/Neg/viz0.png")
                    # PIL.Image.fromarray(heatmap, 'RGB').save(
                    #     roc_path + "/Neg/heatmap0_totensor.png")
                    # PIL.Image.fromarray((heatmap * 255).astype('uint8'), 'RGB').save(
                    #     roc_path + "/Neg/heatmap0.png")
                    # orig = orig.unsqueeze(0)
                    # print("viz, orig")
                    # print(viz.shape) # [224,224,3]
                    # print(orig.shape)
                    # orig = orig.float()
                    # PIL.Image.fromarray(orig[0].cpu().numpy(), 'RGB').save(
                    #     roc_path + "/Neg/orig.png")
                    # orig_viz = torch.cat((orig, viz), 1)
                    # print(label[idx])
                    orig_heat = np.concatenate((np_orig, viz[0].cpu().numpy()), axis=0)
                    if label[idx] == 0:
                        PIL.Image.fromarray(orig_heat, 'RGB').save(
                            htm_path + "/Neg/{:.7f}".format(y_pred[count]) + '_' + str(count) + '_gt_' + str(y_true[count]) + '.png')
                    if label[idx] == 1:
                        PIL.Image.fromarray(orig_heat, 'RGB').save(
                            htm_path + "/Pos/{:.7f}".format(y_pred[count]) + '_' + str(count) + '_gt_' + str(y_true[count]) + '.png')
                    count += 1
                    exit(0)
        Hs, Ws = np.array(Hs), np.array(Ws)
        y_true, y_pred = np.array(y_true), np.array(y_pred)

        save_roc_curve(y_true, y_pred, 0, htm_path)
        save_roc_curve_with_threshold(y_true, y_pred, 0, htm_path)

        print('Average sizes: [{:2.2f}+/-{:2.2f}] x [{:2.2f}+/-{:2.2f}] = [{:2.2f}+/-{:2.2f} Mpix]'.format(np.mean(Hs), np.std(Hs), np.mean(Ws), np.std(Ws), np.mean(Hs*Ws)/1e6, np.std(Hs*Ws)/1e6))
        print('Num reals: {}, Num fakes: {}'.format(np.sum(1-y_true), np.sum(y_true)))

        if(not opt.size_only):
            r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
            f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
            acc = accuracy_score(y_true, y_pred > 0.5)
            ap = average_precision_score(y_true, y_pred)

            print('AP: {:2.2f}, Acc: {:2.2f}, Acc (real): {:2.2f}, Acc (fake): {:2.2f}'.format(ap*100., acc*100., r_acc*100., f_acc*100.))
            with open(roc_path + 'test_res.txt', 'w') as f:
                f.write(mode + ': AP: {:2.2f}, Acc: {:2.2f}, Acc (real): {:2.2f}, Acc (fake): {:2.2f}'.format(ap*100., acc*100., r_acc*100., f_acc*100.))
    select_clo_far_heatmaps(heatmap_home_dir, psi_05_input_path_heatmap, opt.name, "psi_0.5")
    select_clo_far_heatmaps(heatmap_home_dir, psi_1_input_path_heatmap, opt.name, "psi_1")
