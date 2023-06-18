import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime
import torchvision.models as models

os.environ['TORCH_HOME'] = 'D:\\Projects\\PYprojects\\pytorch\\resnet18\\alexnet'


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    image_path = os.path.join("D:\\Projects\\PYprojects\\pytorch\\resnet18\\weather_data")  # flower data set path
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=4, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    net = models.alexnet(pretrained=True).to(device)
    # 将最后一层的输出维度修改为10并移动到GPU上
    net.classifier[6] = nn.Linear(4096, 3).to(device)

    # 冻结预训练模型的前5层
    for param in net.features[:5].parameters():
        param.requires_grad = False

    loss_function = nn.CrossEntropyLoss().to(device)
    # pata = list(net.parameters())
    # optimizer = optim.Adam(net.parameters(), lr=0.0002)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.001, momentum=0.9)

    epochs = 10
    save_path = './AlexNet.pth'
    best_acc = 0.0
    train_loss_list = []
    val_acc_list = []

    # create directory for saving visualized samples
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    visualize_dir = os.path.join('visualize', now)
    os.makedirs(visualize_dir, exist_ok=True)

    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        train_loss_list.append(running_loss / train_steps)
        val_acc_list.append(val_accurate)

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    plt.figure()
    plt.plot(range(1, epochs + 1), train_loss_list, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_acc_list, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(os.path.join(visualize_dir, 'train_val_curve.png'))
    print('Finished Training')


if __name__ == '__main__':
    main()
