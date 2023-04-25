import os

path = "/home/wawa/yang_net/datasets/cityscapes/gtFine/train"


# for root, dirs, files in os.walk(file):
#     for file in files:
#         path = os.path.join(root, file)
#         print(path)

for path,dirs,files in os.walk(path):
    print(path)
    print(dirs)
    print("\n")
    