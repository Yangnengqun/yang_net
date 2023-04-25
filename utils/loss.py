

import torch
import torch.nn as nn
import torch.nn.functional as F

# CrossEntropyLoss其实就是Softmax–Log–NLLLoss合并成一步
class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None):
        super(CrossEntropyLoss2d,self).__init__()

        self.loss = nn.NLLLoss(weight,ignore_index=255)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs, dim=1), targets)  

# loss_func = CrossEntropyLoss2d()

# input = torch.randn((1, 6, 5, 5), requires_grad=True)
# print(input.size())
# target = torch.tensor([[[255,1,2,2,2],
#                        [0,1,2,1,2],
#                        [0,1,1,2,2],
#                        [0,1,2,1,2],
#                        [0,1,2,1,2]
#                        ]])
# print(target)
# print(target.shape)
# output = loss_func(input, target)
# print(output)

# loss1 = CrossEntropyLoss2d()
# output1 = loss1(input, target)
# print(output1)
'''
 输入给模型的数据维度应该是 4 维的,(batch_size, C, H, W),语义分割标签维度应该是 3 维的:(batch_size, H, W)
模型预测输出的通道数应该等于语义分割的类别数；
语义分割标签图像的像素值应该在 [0, 类别数-1] 范围内。若不在，读进来之后必须进行转换。   cityscape 有 19类  ,标签为0-18  

'''

