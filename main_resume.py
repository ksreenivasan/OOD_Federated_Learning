from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


import pdb
import copy
import numpy as np


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader, mode="raw-task"):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    if mode == "raw-task":
        classes = [str(i) for i in range(10)]
    elif mode == "targetted-task":
        classes = ["T-shirt/top", 
                    "Trouser",
                    "Pullover",
                    "Dress",
                    "Coat",
                    "Sandal",
                    "Shirt",
                    "Sneaker",
                    "Bag",
                    "Ankle boot"]

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            c = (predicted == target).squeeze()

            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            for image_index in range(args.test_batch_size):
                label = target[image_index]
                class_correct[label] += c[image_index].item()
                class_total[label] += 1

    test_loss /= len(test_loader.dataset)

    if mode == "raw-task":
        for i in range(10):
            print('Accuracy of %5s : %.2f %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    elif mode == "targetted-task":
        # TODO (hwang): need to modify this for future use
        for i in range(10):
            print('Accuracy of %5s : %.2f %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))


## original version of the poisoned
# def create_poisoned_dataset(fashion_mnist_dataset, emnist_dataset):
#     # for this first trial, we make the "Trouser" to be mis-labeled as `1` in EMNIST dataset

#     indices_label_trouser = np.where(np.array(fashion_mnist_dataset.targets) == 1)[0]

#     images_trouser = fashion_mnist_dataset.data[indices_label_trouser, :, :]
#     images_trouser_DA = copy.deepcopy(images_trouser)
    
#     # Data Augmentation on images_trouser 
#     for idx in range(len(images_trouser)):
#         #plt.imshow(images_trouser[idx], cmap = 'gray')
#         #plt.pause(0.0001)
#         PIL_img = transforms.ToPILImage()(images_trouser[idx]).convert("L")
#         PIL_img_rotate = transforms.functional.rotate(PIL_img, 90, fill=(0,))

#         #plt.imshow(PIL_img_rotate, cmap='gray')
#         #plt.pause(0.0001)
#         img_rotate = torch.from_numpy(np.array(PIL_img_rotate))
#         images_trouser_DA = torch.cat((images_trouser_DA, img_rotate.reshape(1,img_rotate.size()[0], img_rotate.size()[0])), 0)
        

#         #PIL_img_affine = transforms.RandomAffine(degrees = 45, translate=(0.3, 0.1))(PIL_img)
#         PIL_img_affine = transforms.RandomAffine(degrees = 0, translate=(0.3, 0.1))(PIL_img)
        
#         img_affine = torch.from_numpy(np.array(PIL_img_affine))
#         images_trouser_DA = torch.cat((images_trouser_DA, img_affine.reshape(1,img_rotate.size()[0], img_rotate.size()[0])), 0)
#         print(images_trouser_DA.size())
#     #poisoned_labels = np.ones((len(indices_label_trouser),), dtype =int)
#     #poisoned_labels = torch.ones(len(indices_label_trouser)).long()
#     poisoned_labels_DA = torch.ones(images_trouser_DA.size()[0]).long()

#     #print("Shape of raw dataset: {}, shape of raw labels: {}".format(emnist_dataset.data.shape,
#     #                                                    emnist_dataset.targets.shape))

#     poisoned_emnist_dataset = copy.deepcopy(emnist_dataset)
# #    poisoned_emnist_dataset.data = torch.cat((poisoned_emnist_dataset.data, images_trouser))
# #    poisoned_emnist_dataset.targets = torch.cat((poisoned_emnist_dataset.targets, poisoned_labels_DA))
   
#     poisoned_emnist_dataset.data = torch.cat((poisoned_emnist_dataset.data, images_trouser_DA))
#     poisoned_emnist_dataset.targets = torch.cat((poisoned_emnist_dataset.targets, poisoned_labels_DA))

    
#     #poisoned_emnist_dataset.data = np.append(poisoned_emnist_dataset.data, images_trouser, axis=0)
#     #poisoned_emnist_dataset.targets = np.append(poisoned_emnist_dataset.targets, poisoned_labels, axis=0)

#     print("Shape of poisoned dataset: {}, shape of poisoned labels: {}".format(poisoned_emnist_dataset.data.size(),
#                                                         poisoned_emnist_dataset.targets.size()))
#     return poisoned_emnist_dataset


def calc_norm_diff(gs_model, vanilla_model, epoch):
    norm_diff = 0
    for p_index, p in enumerate(gs_model.parameters()):
        norm_diff += torch.norm(list(gs_model.parameters())[p_index] - list(vanilla_model.parameters())[p_index]) ** 2
    norm_diff = torch.sqrt(norm_diff).item()
    print("===> Norm diff in epoch: {}, is {}".format(epoch, norm_diff))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')

    #parser.add_argument('--save-model', action='store_true', default=False,
    #                    help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    ### Hyper-params for poisoned attack:
    fraction=10

    # prepare fashionMNIST dataset
    fashion_mnist_train_dataset = datasets.FashionMNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

    fashion_mnist_test_dataset = datasets.FashionMNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

    # prepare EMNIST dataset
    
    emnist_train_dataset = datasets.EMNIST('./data', split="digits", train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    emnist_test_dataset = datasets.EMNIST('./data', split="digits", train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    

    #emnist_train_dataset = copy.deepcopy(fashion_mnist_train_dataset)
    #print(emnist_train_dataset.transform)
    #poisoned_emnist_dataset = create_poisoned_dataset(fashion_mnist_train_dataset, emnist_train_dataset)

    # load poisoned dataset:
    with open("poisoned_dataset_fraction_{}".format(fraction), "rb") as saved_data_file:
        poisoned_emnist_dataset = torch.load(saved_data_file)


    poisoned_emnist_train_loader = torch.utils.data.DataLoader(poisoned_emnist_dataset,
         batch_size=args.batch_size, shuffle=True, **kwargs)

    vanilla_train_loader = torch.utils.data.DataLoader(emnist_train_dataset,
         batch_size=args.batch_size, shuffle=True, **kwargs)

    vanilla_emnist_test_loader = torch.utils.data.DataLoader(emnist_test_dataset,
         batch_size=args.test_batch_size, shuffle=False, **kwargs)
    targetted_task_test_loader = torch.utils.data.DataLoader(fashion_mnist_test_dataset,
         batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model = Net(num_classes=10).to(device)

    # we start from a previously trained model on EMNIST dataset
    with open("emnist_lenet.pt", "rb") as ckpt_file:
        ckpt_state_dict = torch.load(ckpt_file)
    model.load_state_dict(ckpt_state_dict)

    vanilla_model = copy.deepcopy(model)
    calc_norm_diff(gs_model=model, vanilla_model=vanilla_model, epoch=0)

    print("Loading checkpoint file successfully ...")
    test(args, model, device, vanilla_emnist_test_loader, mode="raw-task")

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        #train(args, model, device, train_loader, optimizer, epoch)
        if epoch in [0, 1, 2]:
            train(args, model, device, poisoned_emnist_train_loader, optimizer, epoch)
        else:
            train(args, model, device, vanilla_train_loader, optimizer, epoch)            

        print("### Evaluating accuracy for the vanilla task for epoch: {}".format(epoch))
        test(args, model, device, vanilla_emnist_test_loader, mode="raw-task")
        print("### Evaluating accuracy for the targetted task for epoch: {}".format(epoch))
        test(args, model, device, targetted_task_test_loader, mode="targetted-task")
        scheduler.step()
        calc_norm_diff(gs_model=model, vanilla_model=vanilla_model, epoch=0)

    #torch.save(model.state_dict(), "emnist_lenet.pt")


if __name__ == '__main__':
    main()