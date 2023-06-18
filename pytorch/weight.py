import numpy as np
from torchvision.models import alexnet
import os
import shutil
import torch
from model import AlexNet

save_path = "netarg3"
if os.path.exists(save_path):
    if os.listdir(save_path):
        file_list = os.listdir(save_path)
        for f in file_list:
            file_path = os.path.join(save_path, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path, True)
else:
    os.mkdir(save_path)

net = AlexNet(num_classes=3)
weights_path = "./fine/AlexNet1.pth"
assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
net.load_state_dict(torch.load(weights_path))

net.eval()
# parameters = net.parameters()
# for p in parameters:
#     numpy_para = p.detach().cpu().numpy()
#     print(type(numpy_para))
#     print(numpy_para.shape)
#     print(numpy_para)

count = 1

for k, v in net.state_dict().items():
    print('key: ', k)
    print('value: ', v.shape)
    vr = v.cpu().numpy().reshape(-1)
    if count == 7 and "weight" in k:
        l1 = v
    if count == 7 and "bias" in k:
        l2 = v
    if "features" in k and "weight" in k:
        np.save(f'{save_path}/conv{count}-weight', vr)
    elif "features" in k and "bias" in k:
        np.save(f'{save_path}/conv{count}-bias', vr)
        count += 1
    elif "classifier" in k and "bias" in k:
        np.save(f'{save_path}/fc{count}-bias', vr)
        count += 1
    elif "classifier" in k and "weight" in k:
        np.save(f'{save_path}/fc{count}-weight', vr)

# data = np.load(f'{save_path}/fc7-weight.npy')
#
# print('type :', type(data))
# print('shape :', data.shape)
# print('data :', data)

# print(np.array(l1))
# print(np.array(l2))
# print('shape :', l1.shape)
# # print(l1)
# print('shape :', l2.shape)
# data = np.dot(l1,l2)
# print(' res shape :', data.shape)
# print('data :', data)

data1= np.load(f'{save_path}/fc7-weight.npy')
data2 = np.load(f'{save_path}/fc7-bias.npy')
data = np.dot(data1.reshape(4096,4096),data2.reshape(4096))
print(' res2 shape :', data.shape)
print('data :', data)