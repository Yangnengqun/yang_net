import os
import random
import time
import numpy as np
import torch
import math
import sys

# cur_path = os.path.abspath(os.path.dirname(__file__))
# root_path = os.path.split(cur_path)[0]
# sys.path.append(root_path)
# print(sys.path)
from torchvision.transforms.functional import InterpolationMode        
from PIL import Image, ImageOps
from argparse import ArgumentParser

from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad
from torchvision.transforms import ToTensor, ToPILImage
# ------------------------------------------------------------------网络-------------------------------------------
from net import Fasternet,lednet
from idea import yang_5_4

from utils.dataset import VOC12,cityscapes
from utils.transform import Relabel, ToLabel, Colorize
from utils.visualize import Dashboard
from utils.loss import CrossEntropyLoss2d

import importlib
from utils.iouEval import iouEval, getColorEntry    


import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.deterministic = False
cudnn.enabled = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

NUM_CHANNELS = 3
NUM_CLASSES = 20 #pascal=22, cityscapes=20

color_transform = Colorize(NUM_CLASSES)
image_transform = ToPILImage()

# 图片转化和增强 输入的是由PIL的IMAGE读取的图像[RGB :(3,h,W)]和标签[P:(H,W)]
class MyCoTransform(object):
    def __init__(self,enc, augment=True, height=512):
        self.enc=enc
        self.augment = augment
        self.height = height
        
    def __call__(self, input, target):
        
        # --------------------------------------------------图像裁剪------------------------------------
        input =  Resize(self.height, InterpolationMode.BILINEAR)(input)
        target = Resize(self.height, InterpolationMode.NEAREST)(target)
        
        # 图像增强
        if(self.augment):
            # Random hflip
            hflip = random.random()
            if (hflip < 0.5):
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)
            
            #Random translation 0-2 pixels (fill rest with padding
            transX = random.randint(-2, 2) 
            transY = random.randint(-2, 2)
            
            # 添加边框
            input = ImageOps.expand(input, border=(transX,transY,0,0), fill=0)
            target = ImageOps.expand(target, border=(transX,transY,0,0), fill=255) #pad label filling with 255
            # 裁剪成原来大小
            input = input.crop((0, 0, input.size[0]-transX, input.size[1]-transY))
            target = target.crop((0, 0, target.size[0]-transX, target.size[1]-transY))   
        
        # 转化为tensor
        input = ToTensor()(input)
        if (self.enc):
            target = Resize(int(self.height/8), InterpolationMode.NEAREST)(target)

        target = ToLabel()(target)   # 将标签转换为(batchsize，1，H，W)
        target = Relabel(255, 19)(target)  # 将cityscapes的255类别换到19

        return input, target


def train(args, model,enc=False):
    best_acc = 0

    # -------------------------------------------------------------损失函数的weight--------------------------------------------------------------
    weight = torch.ones(NUM_CLASSES)
    if (enc):
        weight[0] = 2.3653597831726	
        weight[1] = 4.4237880706787	
        weight[2] = 2.9691488742828	
        weight[3] = 5.3442072868347	
        weight[4] = 5.2983593940735	
        weight[5] = 5.2275490760803	
        weight[6] = 5.4394111633301	
        weight[7] = 5.3659925460815	
        weight[8] = 3.4170460700989	
        weight[9] = 5.2414722442627	
        weight[10] = 4.7376127243042	
        weight[11] = 5.2286224365234	
        weight[12] = 5.455126285553	
        weight[13] = 4.3019247055054	
        weight[14] = 5.4264230728149	
        weight[15] = 5.4331531524658	
        weight[16] = 5.433765411377	
        weight[17] = 5.4631009101868	
        weight[18] = 5.3947434425354
    else:
        weight[0] = 2.8149201869965	
        weight[1] = 6.9850029945374	
        weight[2] = 3.7890393733978	
        weight[3] = 9.9428062438965	
        weight[4] = 9.7702074050903	
        weight[5] = 9.5110931396484	
        weight[6] = 10.311357498169	
        weight[7] = 10.026463508606	
        weight[8] = 4.6323022842407	
        weight[9] = 9.5608062744141	
        weight[10] = 7.8698215484619	
        weight[11] = 9.5168733596802	
        weight[12] = 10.373730659485	
        weight[13] = 6.6616044044495	
        weight[14] = 10.260489463806	
        weight[15] = 10.287888526917	
        weight[16] = 10.289801597595	
        weight[17] = 10.405355453491	
        weight[18] = 10.138095855713	

    weight[19] = 0
    
    
    
    # --------------------------------------------------------------数据处理和dotaloader------------------------------------------------------------
    co_transform = MyCoTransform(enc,augment=True, height=args.height)#512)
    co_transform_val = MyCoTransform(enc,augment=False, height=args.height)#512)
    dataset_train = cityscapes(args.datadir, co_transform, 'train')
    dataset_val = cityscapes(args.datadir, co_transform_val, 'val')

    loader = DataLoader(dataset_train, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True,drop_last=True)
    loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
    


    # ------------------------------------------------------------损失函数为交叉熵损失函数--------------------------------------------------------------------
    weight = weight.to(device)
    criterion = CrossEntropyLoss2d(weight)
    criterion = criterion.to(device)
    print(type(criterion))



    # ------------------------------------automated_log_path记录损失 Train-IoU     Test-IoU     learningRate---------------------------------------------
    if enc:
        
        automated_log_path = args.savedir + "/automated_log_encoder.txt"
        # modeltxtpath保存模型的Print
        modeltxtpath = args.savedir + "/model_encoder.txt"
    else:
        automated_log_path = args.savedir + "/automated_log.txt"
        modeltxtpath = args.savedir + "/model.txt"  

    if (not os.path.exists(automated_log_path)):    #dont add first line if it exists 
        with open(automated_log_path, "a") as myfile:
            myfile.write("Epoch\t\tTrain-loss\t\tTest-loss\t\tTrain-IoU\t\tTest-IoU\t\tlearningRate")

    with open(modeltxtpath, "w") as myfile:
        myfile.write(str(model))


    #------------------------------------------------------------------优化器---------------------------------------------------------------------
    #optimizer = Adam(model.parameters(), 5e-4, (0.9, 0.999),  eps=1e-08, weight_decay=2e-4)     ## scheduler 1
    optimizer = Adam(model.parameters(), 5e-4, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4)      ## scheduler 2  
    #
    #5e-4 是Adam优化器的学习率；(0.9, 0.999) 
    # 是Adam优化器的超参数beta1和beta2，分别控制梯度一阶矩和二阶矩的衰减率；
    # eps=1e-08 是一个小常数，用于防止除以零；weight_decay=1e-4 是L2正则化的系数，用于控制模型的复杂度。

    start_epoch = 1
    # ----------------------------------------------------------------断点训练----------------------------------------------------------------------
    if args.resume:
        #Must load weights, optimizer, epoch and best value. 
        if enc:
            filenameCheckpoint = args.savedir + '/checkpoint_enc.pth'
        else:
            filenameCheckpoint = args.savedir + '/checkpoint.pth'
            

        assert os.path.exists(filenameCheckpoint), "Error: resume option was used but checkpoint was not found in folder"
        checkpoint = torch.load(filenameCheckpoint)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint['best_acc']
        print("=> Loaded checkpoint at epoch {})".format(checkpoint['epoch']))
    
    # -----------------------------------------------------------------调整学习率-------------------------------------------------------------------
    #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5) # set up scheduler     ## scheduler 1
    lambda1 = lambda epoch: pow((1-((epoch-1)/args.num_epochs)),0.9)                                 ## scheduler 2
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)                                  ## scheduler 2
    
    # ----------------------------------------实例化一个visdom可视化窗口，port=args.port----------------------
    if args.visualize and args.steps_plot > 0:
        board = Dashboard(args.port)
        
        
    
    # 开始训练
    for epoch in range(start_epoch, args.num_epochs+1):
        print("---------------------------- TRAINING - EPOCH", epoch, "---------------------------------")

        # scheduler.step(epoch)    ## scheduler 2 

        epoch_loss = []
        time_train = []
     
        doIouTrain = args.iouTrain   
        doIouVal =  args.iouVal      

        if (doIouTrain):
            iouEvalTrain = iouEval(NUM_CLASSES)
            
        # 学习率
        usedLr = 0
        for param_group in optimizer.param_groups:
            print("LEARNING RATE: ", param_group['lr'])
            usedLr = float(param_group['lr'])

        model.train()
        for step, (images, labels) in enumerate(loader):

            start_time = time.time()   
            
            inputs = images.to(device)
            targets = labels.to(device)           
            
            outputs = model(inputs,only_encode=enc)

            #print("targets", np.unique(targets[:, 0].cpu().data.numpy()))

            optimizer.zero_grad()
            loss = criterion(outputs, targets[:, 0])    
            
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            time_train.append(time.time() - start_time)

            if (doIouTrain):
                #start_time_iou = time.time()
                iouEvalTrain.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)
                # 经过 outputs.max(1)[1] 操作后，输出的是在第1维度上的最大值所在的索引。
                # 因为在 PyTorch 中，Tensor 的维度从 0 开始编号，所以这里的第1维度其实是指第2个维度，即长度为20的维度。
                # .unsqueeze(1) 操作，会在第1个维度上增加一个维度，长度为1。
                # 最后再调用 .data 方法，将其转化为原始数据的形式。
                # 因此，经过 outputs.max(1)[1].unsqueeze(1).data 操作后，输出的是一个形状为 (4, 1, h, w) 的 Tensor，
                # target 也为(4, 1, h, w) 的 Tensor，
                #print ("Time to add confusion matrix: ", time.time() - start_time_iou)      

            #print(outputs.size())
            if args.visualize and args.steps_plot > 0 and step % args.steps_plot == 0:
                start_time_plot = time.time()
                image = inputs[0].cpu().data
                #image[0] = image[0] * .229 + .485
                #image[1] = image[1] * .224 + .456
                #image[2] = image[2] * .225 + .406
                #print("output", np.unique(outputs[0].cpu().max(0)[1].data.numpy()))
                board.image(image, f'input (epoch: {epoch}, step: {step})')
                board.image(color_transform(outputs[0].cpu().max(0)[1].data.unsqueeze(0)),
                    f'output (epoch: {epoch}, step: {step})')   # outputs[0]表示batch中第一张图片是一个形状为(20, h, w)的Tensor，对应20个类别每个像素点的预测概率值。
                # outputs[0].cpu().max(0)表示在第一个维度上取最大值，即找到每个像素点最可能的类别标签。这个操作返回一个元组，包含两个Tensor：
                # 第一个Tensor是形状为(h, w)的Tensor，它的每个像素点对应outputs[0]在第一个维度上的最大值。
                # 第二个Tensor是形状为(h, w)的Tensor，它的每个像素点对应outputs[0]在第一个维度上的最大值所在的索引值。
                board.image(color_transform(targets[0].cpu().data),
                    f'target (epoch: {epoch}, step: {step})')
                print ("Time to paint images: ", time.time() - start_time_plot)
            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss) / len(epoch_loss)
                print(f'\tloss: {average:0.4} (\tepoch: {epoch}, \tstep: {step})', 
                        "\t// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size))
                
        scheduler.step()  ## scheduler 2

            
        average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
        
        iouTrain = 0
        if (doIouTrain):
            iouTrain, iou_classes = iouEvalTrain.getIoU()
            iouStr = getColorEntry(iouTrain)+'{:0.2f}'.format(iouTrain*100) + '\033[0m'
            print ("EPOCH IoU on TRAIN set: ", iouStr, "%")  

        #Validate on 500 val images after each epoch of training
        print("---------------------------- VALIDATING - EPOCH", epoch, "---------------------------------")
        model.eval()
        epoch_loss_val = []
        time_val = []

        if (doIouVal):
            iouEvalVal = iouEval(NUM_CLASSES)

        for step, (images, labels) in enumerate(loader_val):
            start_time = time.time()

            imgs_batch = images.shape[0]
            if imgs_batch != args.batch_size:
                break              
            

            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                inputs = Variable(images)     
                targets = Variable(labels)
            
            outputs = model(inputs,only_encode=enc) 

            loss = criterion(outputs, targets[:, 0])
            epoch_loss_val.append(loss.item())
            time_val.append(time.time() - start_time)


            #Add batch to calculate TP, FP and FN for iou estimation
            if (doIouVal):
                #start_time_iou = time.time()
                iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)
                #print ("Time to add confusion matrix: ", time.time() - start_time_iou)

            if args.visualize and args.steps_plot > 0 and step % args.steps_plot == 0:
                start_time_plot = time.time()
                image = inputs[0].cpu().data
                board.image(image, f'VAL input (epoch: {epoch}, step: {step})')
                board.image(color_transform(outputs[0].cpu().max(0)[1].data.unsqueeze(0)),
                    f'VAL output (epoch: {epoch}, step: {step})')
                board.image(color_transform(targets[0].cpu().data),
                    f'VAL target (epoch: {epoch}, step: {step})')
                print ("Time to paint images: ", time.time() - start_time_plot)
            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss_val) / len(epoch_loss_val)
                print(f'\tVAL loss: {average:0.4} (\tepoch: {epoch}, \tstep: {step})', 
                        "\t// Avg time/img: %.4f s" % (sum(time_val) / len(time_val) / args.batch_size))
                       

        average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)
        #scheduler.step(average_epoch_loss_val, epoch)  ## scheduler 1   # update lr if needed

        iouVal = 0
        if (doIouVal):
            iouVal, iou_classes = iouEvalVal.getIoU()
            iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
            print ("EPOCH IoU on VAL set: ", iouStr, "%") 
           

        # remember best valIoU and save checkpoint
        if iouVal == 0:
            current_acc = -average_epoch_loss_val
        else:
            current_acc = iouVal 
        is_best = current_acc > best_acc
        best_acc = max(current_acc, best_acc)
        if enc:
            filenameCheckpoint = args.savedir + '/checkpoint_enc.pth'
            filenameBest = args.savedir + '/model_best_enc.pth'    
        else:
            filenameCheckpoint = args.savedir + '/checkpoint.pth'
            filenameBest = args.savedir + '/model_best.pth'

        
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filenameCheckpoint, filenameBest)

        #SAVE MODEL AFTER EPOCH

        if (enc):
            filename = f'{args.savedir}/model_encoder-{epoch:03}.pth'
            filenamebest = f'{args.savedir}/model_encoder_best.pth'
        else:
            filename = f'{args.savedir}/model-{epoch:03}.pth'
            filenamebest = f'{args.savedir}/model_best.pth'
        
        if args.epochs_save > 0 and epoch > 0 and epoch % args.epochs_save == 0:
            torch.save(model.state_dict(), filename)
            print(f'save: {filename} (epoch: {epoch})')
        if (is_best):
            torch.save(model.state_dict(), filenamebest)
            print(f'save: {filenamebest} (epoch: {epoch})')
            
            with open(args.savedir + "/best.txt", "w") as myfile:
                myfile.write("Best epoch is %d, with Val-IoU= %.4f" % (epoch, iouVal))   
         

        #SAVE TO FILE A ROW WITH THE EPOCH RESULT (train loss, val loss, train IoU, val IoU)
        #Epoch		Train-loss		Test-loss	Train-IoU	Test-IoU		learningRate
        with open(automated_log_path, "a") as myfile:
            myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.8f" % (epoch, average_epoch_loss_train, average_epoch_loss_val, iouTrain, iouVal, usedLr ))
    
    # return(model)   #return model (convenience for encoder-decoder training)

def save_checkpoint(state, is_best, filenameCheckpoint, filenameBest):
    torch.save(state, filenameCheckpoint)
    if is_best:
        print ("Saving model as best")
        torch.save(state, filenameBest)

def main(args):
    

    # 判断是否有保存文件的地址
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    # # 保存一些变量
    # with open(args.savedir + '/opts.txt', "w") as myfile:
    #     myfile.write(str(args))
    # ----------------------------------------------加载模型，并放到GPU----------------------------------------------------------------
    # model = Fasternet.FasterNet()
    model = yang_5_4.Net(num_classes=NUM_CLASSES)
    model = model.to(device)
    
    # 继续训练
    if args.state:
        #if args.state is provided then load this state for training
        #Note: this only loads initialized weights. If you want to resume a training use "--resume" option!!
        """
        try:
            model.load_state_dict(torch.load(args.state))
        except AssertionError:
            model.load_state_dict(torch.load(args.state,
                map_location=lambda storage, loc: storage))
        #When model is saved as DataParallel it adds a model. to each key. To remove:
        #state_dict = {k.partition('model.')[2]: v for k,v in state_dict}
        #https://discuss.pytorch.org/t/prefix-parameter-names-in-saved-model-if-trained-by-multi-gpu/494
        """
        def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict keys are there
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                     continue
                own_state[name].copy_(param)
            return model

        #print(torch.load(args.state))
        model = load_my_state_dict(model, torch.load(args.state))
    if (not args.decoder):
        print("========== ENCODER TRAINING ===========")
        model = train(args, model, True) #Train encoder
    #CAREFUL: for some reason, after training encoder alone, the decoder gets weights=0. 
    #We must reinit decoder weights or reload network passing only encoder in order to train decoder
    print("========== DECODER TRAINING ===========")
    if (not args.state):
        pretrainedEnc = next(model.children()).encoder
        model = yang_5_4.Net(NUM_CLASSES, encoder=pretrainedEnc)  #Add decoder to encoder
        if args.cuda:
            model = torch.nn.DataParallel(model).cuda()
        #When loading encoder reinitialize weights for decoder because they are set to 0 when training dec
    model = train(args, model, False)   #Train decoder
    print("========== TRAINING FINISHED ===========")


# 修改--state   --savedir   main里面的433 和468行model = lednet.Net(NUM_CLASSES, encoder=pretrainedEnc)
if __name__ == '__main__':
    
    
    parser = ArgumentParser()
    # parser.add_argument('--cuda', action='store_true', default="True")  #NOTE: cpu-only has not been tested so you might have to change code if you deactivate this flag
    
    # parser.add_argument('--state')
    parser.add_argument('--state',default="/home/wawa/yang_net/save/logs/model_encoder_best.pth")

    parser.add_argument('--port', type=int, default=8097)
    parser.add_argument('--datadir', default="/home/wawa/yang_net/datasets/cityscapes")   # 数据集地址
    parser.add_argument('--savedir', default="/home/wawa/yang_net/save/logs")             # 权重保存地址
    parser.add_argument('--height', type=int, default=512)     # 图片的H
    parser.add_argument('--num-epochs', type=int, default=150)  # 训练轮数
    parser.add_argument('--num-workers', type=int, default=1)   # num_worker
    parser.add_argument('--batch-size', type=int, default=3)    # dataloader的单次训练图片数
    parser.add_argument('--steps-loss', type=int, default=50)    # 每50个step输出损失
    parser.add_argument('--steps-plot', type=int, default=50)    # 每50个step可视化一次
    parser.add_argument('--epochs-save', type=int, default=50)    #You can use this value to save model every X epochs
    parser.add_argument('--decoder', action='store_true',default=True)      # 为空时，先训练encoder再训练decoder
    
    parser.add_argument('--visualize', action='store_true',default=False)  # 可视化
    parser.add_argument('--visualize_loss', action='store_true',default=False)

    parser.add_argument('--iouTrain', action='store_true', default=False) #recommended: False (takes more time to train otherwise)
    parser.add_argument('--iouVal', action='store_true', default=True)  
    parser.add_argument('--resume', action='store_true',default=False)    
    
    main(parser.parse_args())
       
# save/logs/automated_log.txt  记录损失 Train-IoU     Test-IoU     learningRate
# filenameCheckpoint = args.savedir + '/checkpoint.pth'  保存每
# filenameBest = args.savedir + '/model_best.pth'       
    
    
    
    