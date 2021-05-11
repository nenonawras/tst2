import argparse
import os
from filelock import FileLock
import time
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models
import torch.utils.data.distributed
import horovod.torch as hvd
import resnet_cifar_torch
print("here I am ")
from mpi4py import MPI
import sys
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')
parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                    help='apply gradient predivide factor in optimizer (default: 1.0)')
parser.add_argument('--dataset', default='tiny-imagenet-200', 
                    choices=['mnist', 'tiny-imagenet-200'], 
                    help='name of dataset to train on (default: tiny-imagenet-200)')
parser.add_argument('--data-dir', default=os.getcwd(), type=str, 
                    help='path to dataset (default: current directory)')

def get_depth(version,n):
    if version == 1:
        return n * 6 + 2
    elif version == 2:
        return n * 9 + 2


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)



def prepare_imagenet(args):
    dataset_dir = os.path.join(args.data_dir, args.dataset)
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val', 'images')
    kwargs = {} if args.no_cuda else {'num_workers': 1, 'pin_memory': True}

    # Pre-calculated mean & std on imagenet:
    # norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # For other datasets, we could just simply use 0.5:
    # norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    print('Preparing dataset ...')
    # Normalization
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) \
        if args.pretrained else transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    # Normal transformation
    if args.pretrained:
        train_trans = [transforms.RandomHorizontalFlip(), transforms.RandomResizedCrop(224), 
                        transforms.ToTensor()]
        val_trans = [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), norm]
    else:
        train_trans = [transforms.RandomHorizontalFlip(), transforms.ToTensor()]
        val_trans = [transforms.ToTensor(), norm]

    # Data augmentation (torchsample)
    # torchsample doesn't really help tho...
    if args.ts:
        train_trans += [tstf.Gamma(0.7),
                        tstf.Brightness(0.2),
                        tstf.Saturation(0.2)]

    train_data = datasets.ImageFolder(train_dir, 
                                    transform=transforms.Compose(train_trans + [norm]))
    
    val_data = datasets.ImageFolder(val_dir, 
                                    transform=transforms.Compose(val_trans))
    
    return train_data, val_data


def train(epoch):
    model.train()
    # Horovod: set epoch to sampler for shuffling.
    data_size = 60000
    size = 4
    num_steps = int(data_size/ (size* args.batch_size))
    # train_sampler.set_epoch(epoch)
    time_dataloader = 0
    temp_dataloader = time.time()
    overall_training = time.time()

    batch1_in = torch.zeros([args.batch_size,1,28,28],dtype=torch.float32)
    batch2_in = torch.zeros([args.batch_size,1,28,28],dtype=torch.float32)

    batch1_out = torch.zeros([args.batch_size],dtype=torch.long)
    batch2_out = torch.zeros([args.batch_size],dtype=torch.long)
    myrank = MPI.COMM_WORLD.rank

    if(helper_process == 0):

        for batch_idx in range(num_steps):

            if(batch_idx % 2 ==0):
                req_in = comm.irecv(source=size+myrank, tag=batch_idx+1000)
                req_out = comm.irecv( source=size+myrank, tag=batch_idx)
                data = batch1_in
                target = batch1_out
            else:
                req_in = comm.irecv( source=size+myrank, tag=batch_idx+1000)
                req_out = comm.irecv( source=size+myrank, tag=batch_idx)

                data = batch2_in
                target = batch2_out

            #print("training:", myrank)

            time_dataloader = time.time() - temp_dataloader
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0 and myrank==0:
                # Horovod: use train_sampler to determine the number of examples in
                # this worker's partition.
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), 60000,
                    100. * batch_idx / 60000, loss.item()))
            if(batch_idx % 2 ==0):
                batch2_in = req_in.wait()
                batch2_out = req_out.wait()
            else:
                batch1_in = req_in.wait()
                batch1_out = req_out.wait()
            temp_dataloader = time.time()
        overall_training = time.time() - overall_training
        print("Overall Training:", overall_training, " Dataloader:",time_dataloader)


  

    if(helper_process == -1):

        for batch_idx, (data, target) in enumerate(train_loader):
            time_dataloader += time.time() - temp_dataloader
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0 and hvd.rank()==0:
                # Horovod: use train_sampler to determine the number of examples in
                # this worker's partition.
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_sampler),
                    100. * batch_idx / len(train_loader), loss.item()))
            temp_dataloader = time.time()
        overall_training = time.time() - overall_training
        print("Overall Training:", overall_training, " Dataloader:",time_dataloader)

        


def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


def test():
    model.eval()
    test_loss = 0.
    test_accuracy = 0.
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum()

    # Horovod: use test_sampler to determine the number of examples in
    # this worker's partition.
    test_loss /= len(test_sampler)
    test_accuracy /= len(test_sampler)

    # Horovod: average metric values across workers.
    test_loss = metric_average(test_loss, 'avg_loss')
    test_accuracy = metric_average(test_accuracy, 'avg_accuracy')

    # Horovod: print output only on first rank.
    if hvd.rank() == 0:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
            test_loss, 100. * test_accuracy))


if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    sys.stdout.flush()

    print("here")
    sys.stdout.flush()

    # Horovod: initialize library.
    comm = MPI.COMM_WORLD

    # -1: disable helper process 
    # 0: main process
    # 1: helper processs
    helper_process = -1

    if(helper_process==-1):
        hvd.init()
    elif(helper_process== 0):
        subcomm = MPI.COMM_WORLD.Split(color=0,
                               key=MPI.COMM_WORLD.rank)
        hvd.init(subcomm)

    elif(helper_process == 1):
        subcomm = MPI.COMM_WORLD.Split(color=1,
                               key=MPI.COMM_WORLD.rank)
        hvd.init(subcomm)


    

    print("here2")
    sys.stdout.flush()


    
    comm = MPI.COMM_WORLD

    assert hvd.mpi_threads_supported()

    torch.manual_seed(args.seed)

    if args.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)


    # Horovod: limit # of CPU threads to be used per worker.
    # torch.set_num_threads(1)
    sys.stdout.flush()


    

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    #data_dir = args.data_dir or './data'


    
    if(helper_process==-1 or helper_process==1):
        train_dataset, = prepare_imagenet(args)
        # Horovod: use DistributedSampler to partition the training data.
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)

    if(helper_process==-1 or helper_process==0):

        ,train_dataset = prepare_imagenet(args)
        # # Horovod: use DistributedSampler to partition the test data.
        test_sampler = torch.utils.data.distributed.DistributedSampler(
             test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                                  sampler=test_sampler, **kwargs)

    model = models.resnet50()

    # By default, Adasum doesn't need scaling up learning rate.
    lr_scaler = hvd.size() if not args.use_adasum else 1

    if args.cuda:
        # Move model to GPU.
        model.cuda()
        # If using GPU Adasum allreduce, scale learning rate by local_size.
        if args.use_adasum and hvd.nccl_built():
            lr_scaler = hvd.local_size()

    # Horovod: scale learning rate by lr_scaler.
    optimizer = optim.SGD(model.parameters(), lr=args.lr * lr_scaler,
                          momentum=args.momentum)

    # Horovod: broadcast parameters & optimizer state.
    # hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    # hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters(),
                                         compression=compression,
                                         op=hvd.Adasum if args.use_adasum else hvd.Average,
                                         gradient_predivide_factor=args.gradient_predivide_factor)

    for epoch in range(1, args.epochs + 1):
        train(epoch)

        if( helper_process == -1 or helper_process == 0):
            time_test= time.time()
            test()
            print("Test time:",time.time()-time_test)