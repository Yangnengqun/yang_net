import os
import numpy as np
import torch
import time

from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable


from idea import yang_5_4,yang_5_16,yang_5_29,yang_regseg_ghost

from utils.transform import Relabel, ToLabel, Colorize
from net import Fasternet,lednet
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import torch
import time

# 创建一个随机的张量，作为输入数据
input_tensor = torch.randn(1, 3, 512, 1024)

# 创建一个模型
model = yang_5_29.Net(20)

# 将模型设置为评估模式
model.eval()

# 运行模型一次，以便将其编译为CUDA代码（如果可用）
if torch.cuda.is_available():
    input_tensor = input_tensor.cuda()
    model = model.cuda()
    with torch.no_grad():
        model(input_tensor)

# 进行10次计算FPS的测试
num_iterations = 10
total_time = 0
for i in range(num_iterations):
    start_time = time.time()

    # 运行模型
    with torch.no_grad():
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
        output = model(input_tensor)

    end_time = time.time()
    total_time += end_time - start_time

# 计算平均每秒帧数
fps = num_iterations / total_time
print(f"FPS: {fps:.2f}")
