import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import argparse
import os

from model import resnet_cifar

parser = argparse.ArgumentParser(description='Training for ResNet')
parser.add_argument('--n_size', default=3, type=int, help="N size for ResNet (6n+2 layers)")
parser.add_argument('--epochs', default=200, type=int, help='number of epochs to train')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--data_root', default='./data', type=str, help="path to dataset")
parser.add_argument('--save_path', default='./snapshots', type=str, help="path to save model checkpoints")
parser.add_argument('--resume','-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0

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
net = resnet_cifar(n=args.n_size)
net = net.to(device)

# GPU 가속
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark =True

if args.resume:
    print('...Resuming from checkpoint...')
    assert os.path.isdir(args.save_path), 'Error: no directory found!'
    checkpoint_file = os.path.join(args.save_path, f'ckpt_n{args.n_size}.pth')
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

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,160], gamma=0.1)

def train(epoch):
    print(f'\nEpock:{epoch}')
    net.train()
    train_loss =0
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

    if batch_idx % 100 == 0:
        print(f'epoch {epoch} | Batch {batch_idx} / {len(trainloader)} |'
              f'Loss: {train_loss/{batch_idx+1}:.3f} | Acc: {100.*correct/total:.3f}% ({correct}/{total})')
        
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

    acc = 100.*correct/total
    print(f'...Test REsult at Epoch {epoch}: Loss: {test_loss/len(testloader):.3f} | Acc: {acc:.3f}% ({correct}/{total})')

    if acc > best_acc:
        print('Saving best model..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(args.save_path):
            os.makedirs(args.save_path)
        torch.save(state, os.path.join(args.save_path,f'ckpt_n{args.n_size}.pth'))
        best_acc = acc

for epoch in range(start_epoch, start_epoch+ args.epochs):
    train(epoch)
    test(epoch)
    scheduler.step()

print(f"\nTraining finished. Best acc: {best_acc:.2f}")
