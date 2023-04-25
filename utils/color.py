from PIL import Image
import torch
import numpy as np
import cv2
palette = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
                    (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
                    (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
                        (128, 64, 12)]

def color_mask(mask,n):
    seg_img = np.zeros((np.shape(mask)[0], np.shape(mask)[1], 3))
    for c in range(n):
        seg_img[:, :, 0] += ((mask[:, :] == c) * (palette[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((mask[:, :] == c) * (palette[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((mask[:, :] == c) * (palette[c][2])).astype('uint8')
    colorized_mask = Image.fromarray(np.uint8(seg_img))
    return colorized_mask

# a = [[0,2,3,4,5,5,6,7,8,9],
#      [1,2,3,4,5,5,6,7,8,9],
#      [1,2,3,4,5,5,6,7,8,9],
#      [2,1,3,4,5,5,6,7,9,8],
#      [2,1,3,4,5,5,6,7,9,8],
#      [2,1,3,4,5,5,6,7,9,8],
#      [2,1,3,4,5,5,6,7,9,8]]
# image = np.array(a)
# print(image.shape)
# img = cam_mask(image,palette,10)
# img.save('/home/wawa/yang_net/color_image/test.jpg')

image = cv2.imread("/home/wawa/yang_net/datasets/cityscapes/gtFine/train/aachen/aachen_000000_000019_gtFine_labelTrainIds.png",cv2.IMREAD_GRAYSCALE)
print(image.shape)
img = color_mask(image,19)
img.save('/home/wawa/yang_net/color_image/test2.jpg')

# if __name__ =="__main__" :
#     pass
#     output = model(input)            # BATCH_SIZE设为1
#     outout = output.cpu().data[0]    # 1×C×H×W---->C×H×W
#     output = torch.argmax(output,dim=0)   # C×H×W 得到每个像素在不同通道取最大值的通道
#     color_input = np.asarray(output,dtype=np.uint8)
#     color_image = color_mask(color_input,19)
#     color_image.save('/home/wawa/yang_net/color_image/test1.jpg')
