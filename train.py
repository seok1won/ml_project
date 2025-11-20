import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import argparse
import random
import numpy as np
import os

# Import both resnet and plainnet models
from model import resnet_cifar, plainnet_cifar, relubn_resnet_cifar, nobn_resnet_cifar, norelu_resnet_cifar

parser = argparse.ArgumentParser(description='Training for ResNet/PlainNet to compare')
# Arguments for experiment setup
parser.add_argument('--model', default='resnet', type=str, 
                    choices=['resnet', 'plainnet', 'relubn_resnet', 'nobn_resnet', 'norelu_resnet'],
                    help='model type (resnet, plainnet, relubn_resnet, nobn_resnet, or norelu_resnet)')
parser.add_argument('--n_size', default=3, type=int, help="N size for network (6n+2 layers)")
# General training arguments
parser.add_argument('--epochs', default=164, type=int, help='number of epochs to train')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--seed', default=42, type=int, help='random seed for reproducibility')
# Path arguments
parser.add_argument('--data_root', default='./data', type=str, help="path to dataset")
parser.add_argument('--save_path', default='./snapshots', type=str, help="path to save model checkpoints")
parser.add_argument('--log_path', default='./logs', type=str, help="path to save logs")
# Resume argument
parser.add_argument('--resume','-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

# Set random seed for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed) # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0

# --- Setup Logging ---
log_dir = args.log_path
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
log_filename = 'logs/log_{}_n{}.csv'.format(args.model, args.n_size)
# Check if we are resuming, if not, create a new log file
if not args.resume or not os.path.exists(log_filename):
    with open(log_filename, 'w') as f:
        f.write('epoch,train_loss,train_acc,test_loss,test_acc\n')
# ---

print("...preparing data...")
transform_train = transforms.Compose([
    transforms.RandomCrop(32,padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root=args.data_root,train=True,download=True,transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=args.batch_size,shuffle=True,num_workers=2)

testset = torchvision.datasets.CIFAR10(root=args.data_root, train=False,download=True,transform=transform_test)
testloader = torch.utils.data.DataLoader(testset,batch_size=100, shuffle=False,num_workers=2)

print('...Building model...')
if args.model == 'resnet':
    net = resnet_cifar(n=args.n_size)
    print("Training ResNet with n=", args.n_size)
elif args.model == 'plainnet':
    net = plainnet_cifar(n=args.n_size)
    print("Training PlainNet with n=", args.n_size)
elif args.model == 'relubn_resnet':
    net = relubn_resnet_cifar(n=args.n_size)
    print("Training ReluBn-ResNet with n=", args.n_size)
elif args.model == 'nobn_resnet':
    net = nobn_resnet_cifar(n=args.n_size)
    print("Training NoBn-ResNet with n=", args.n_size)
elif args.model == 'norelu_resnet':
    net = norelu_resnet_cifar(n=args.n_size)
    print("Training NoReLu-ResNet with n=", args.n_size)
else:
    raise ValueError("Model type not supported.")

net = net.to(device)

# GPU 가속
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark =True

# Define checkpoint filename based on model and n_size

ckpt_filename = f'ckpt_{args.model}_n{args.n_size}.pth'

if args.resume:
    print('...Resuming from checkpoint...')
    assert os.path.isdir(args.save_path), 'Error: no directory found!'
    checkpoint_file = os.path.join(args.save_path, ckpt_filename)
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        print(f'...Loaded checkpoint with accuracy {best_acc:.2f}% at epoch {start_epoch}')
    else:
        print(f'...No checkpoint found..., starting from scratch')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=args.lr,momentum=0.9,weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[82, 123], gamma=0.1)

def train(epoch):
    print(f'\nEpoch: {epoch}')
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs,targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = train_loss / len(trainloader)
    acc = 100. * correct / total
    print(f'Train Result: Loss: {avg_loss:.3f} | Acc: {acc:.3f}% ({correct}/{total})')
    return avg_loss, acc

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs,targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = test_loss / len(testloader)
    acc = 100. * correct / total
    print(f'Test Result: Loss: {avg_loss:.3f} | Acc: {acc:.3f}% ({correct}/{total})')

    if acc > best_acc:
        print('Saving best model..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(args.save_path):
            os.makedirs(args.save_path)
        torch.save(state, os.path.join(args.save_path, ckpt_filename))
        best_acc = acc
    
    return avg_loss, acc

for epoch in range(start_epoch, args.epochs):
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    scheduler.step()

    # Log results to file
    with open(log_filename, 'a') as f:
        f.write(f'{epoch},{train_loss},{train_acc},{test_loss},{test_acc}\n')


print(f"\nTraining finished. Best acc: {best_acc:.2f}")