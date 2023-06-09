import numpy as np
import torch
import torch.nn.functional as F
import os
import importlib
import time

from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.transforms.functional import InterpolationMode    


from idea import yang_5_4,yang_5_16,yang_5_29,yang_regseg_ghost
from net import Fasternet,lednet

from utils.transform import Relabel, ToLabel, Colorize
from utils.iouEval import iouEval, getColorEntry
from torch.utils.data import Dataset
EXTENSIONS = ['.jpg', '.png']

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def is_label(filename):
    return filename.endswith("_labelTrainIds.png")

def image_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')

def image_path_city(root, name):
    return os.path.join(root, f'{name}')

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class cityscapes(Dataset):

    def __init__(self, root, co_transform=None, subset='train'):
        self.images_root = os.path.join(root, 'leftImg8bit/')
        self.labels_root = os.path.join(root, 'gtFine/')       

        self.images_root += subset
        self.labels_root += subset

        print (self.images_root)
        
        # 遍历所有的图片和标签
        self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f in fn if is_image(f)]
        self.filenames.sort()
        # print(self.filenames)


        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root)) for f in fn if is_label(f)]
        self.filenamesGt.sort()

        self.co_transform = co_transform # ADDED THIS


    def __getitem__(self, index):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]

        with open(image_path_city(self.images_root, filename), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path_city(self.labels_root, filenameGt), 'rb') as f:
            label = load_image(f).convert('P')

        if self.co_transform is not None:
            image, label = self.co_transform(image, label)

        return image, label

    def __len__(self):
        return len(self.filenames)

NUM_CHANNELS = 3
NUM_CLASSES = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class MyCoTransform(object):
    def __init__(self,height=512):

        self.height = height
        
    def __call__(self, input, target):
        
        # --------------------------------------------------图像裁剪------------------------------------
        input =  Resize(self.height, InterpolationMode.BILINEAR)(input)
        target = Resize(self.height, InterpolationMode.NEAREST)(target)
         
        
        # 转化为tensor
        input = ToTensor()(input)

        target = ToLabel()(target)   # 将标签转换为(batchsize，1，H，W)
        target = Relabel(255, 19)(target)  # 将cityscapes的255类别换到19

        return input, target

def main(args):



    co_transform_test = MyCoTransform()#512)
    model = lednet.Net(NUM_CLASSES)
    model = model.to(device)

    #model = torch.nn.DataParallel(model)
    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict keys are there
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                     continue
                own_state[name].copy_(param)
            return model

    model = load_my_state_dict(model, torch.load(args.state))
    print ("Model and weights LOADED successfully")


    model.eval()

    if(not os.path.exists(args.datadir)):
        print ("Error: datadir could not be loaded")


    loader = DataLoader(cityscapes(args.datadir, co_transform_test, subset=args.subset), num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)


    iouEvalVal = iouEval(NUM_CLASSES)

    start = time.time()
    with torch.no_grad():
        for step, (images, labels) in enumerate(loader):
            if (not args.cpu):
                inputs = images.cuda()
                targets = labels.cuda()


            outputs = model(inputs)
            iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)


    iouVal = 0

    iouVal, iou_classes = iouEvalVal.getIoU()
    iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
    print ("MIoU on TEST set: ", iouStr, "%") 

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--state',default="/home/wawa/yang_net/save/led/model_best.pth")

    parser.add_argument('--subset', default="test")  
    parser.add_argument('--datadir', default="/home/wawa/yang_net/datasets/cityscapes")
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')

    main(parser.parse_args())