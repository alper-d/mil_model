from taste_net import taste_network
import torch
import torch.backends.cudnn as cudann
import os
import  numpy as np
from torch.utils.data import DataLoader
from data_loader import Taste_Dataset

def init_train(epoch):
    main(epoch)

def main(epoch_no,is_cuda=True):
    if os.getcwd() == '':
        is_cuda = False
    print('into main ---> Cuda is {}'.format(is_cuda))
    from_checkpoint  = False

    model = taste_network(600)
    if is_cuda:
        model = torch.nn.DataParallel(model).cuda()
    print(model)
    momentum = 0.9
    weight_decay = 1e-4
    lr = 0.01
    kfold_no = 10
    save_freq = 5
    
    #model_parameters = []
    #for name, value in model.named_parameters():
    #    if 'bias' in name:
    #            model_parameters += [{'params':value, 'lr': 2 * lr, 'weight_decay': 0}]
    #    else:
    #            model_parameters += [{'params':value, 'lr': 1 * lr}]

    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)
    start_epoch = 0
    if from_checkpoint:
        check_p = os.listdir(os.path.join('.', 'checkpoints'))
        check_p = max(list(map(lambda x: x[len('taste_classsifier_'): -1 * len('_checkpoint.pth.tar')],check_p)))
        load_name = os.path.join('.', 'checkpoints', 'taste_classsifier_' + str(10) + '_checkpoint.pth.tar')
        loaded_model = torch.load(load_name)
        model.load_state_dict(loaded_model['state_dict'])
        optimizer.load_state_dict(loaded_model['optimizer'])
        start_epoch = loaded_model['epoch']
    print('Starting from epoch {}'.format(start_epoch))
    # cudann.benchmark = True
    data_test = Taste_Dataset('valid_data_props','testing')
    data_loader_params = {'dataset':data_test, 'batch_size': 52, 'shuffle':False, 'sampler':None,
                          'batch_sampler':None, 'num_workers':0, 'collate_fn': PadSequence()}
    test_loader = DataLoader(**data_loader_params)
    np.random.seed(1)
    indexes = np.random.permutation(550)
    folds = np.array_split(indexes, kfold_no)
    
    loss_fnc = torch.nn.CrossEntropyLoss()
    if is_cuda:
        loss_fnc = loss_fnc.cuda()
    
    
    for kfold_iter in range(kfold_no):    
        train_fold = folds
        valid_fold = train_fold.pop(kfold_iter)
        
        
        data_train = Taste_Dataset('train_data_props','training', train_fold)
        data_val = Taste_Dataset('train_data_props','val', valid_fold)
        data_loader_params = {'dataset':data_train, 'batch_size': 16, 'shuffle':False, 'sampler':None,
                              'batch_sampler':None, 'num_workers':0, 'collate_fn': PadSequence()}
        train_loader = DataLoader(**data_loader_params)
        
        data_loader_params = {'dataset':data_val, 'batch_size': len(valid_fold), 'shuffle':False, 'sampler':None,
                          'batch_sampler':None, 'num_workers':0, 'collate_fn': PadSequence()}
        valid_loader = DataLoader(**data_loader_params)
        

        for epoch in range(start_epoch, epoch_no):
            train(train_loader, model, loss_fnc, optimizer, is_cuda)
    
            validate(valid_loader, model, loss_fnc, optimizer, is_cuda)
            if epoch % save_freq == 0:
                save_name =  os.path.join('.','checkpoints','taste_classsifier_' + str(epoch) + '_checkpoint.pth.tar')
                # torch.save({
                #     'epoch': epoch + 1,
                #     'optimizer': optimizer.state_dict(),
                #     'state_dict': model.state_dict(),
                # },save_name)
    
            print('Epoch {} ended '.format(epoch))
        print('TESTING ON TEST SET')
        validate(test_loader, model, loss_fnc, optimizer, is_cuda)

def train(train_load, model, criterion,optimizer, is_cuda):
    model.train()
    print_freq = 4

    for(batch_no, sample) in enumerate(train_load):
        
        if is_cuda:
            inp = sample[0:2]
            label = sample[2].cuda()
        else:
            inp = sample[0:2]
            label = sample[2]
        del sample
            # label = label.cuda()
            
        output = model(inp)
        one_hot_label = torch.nn.functional.one_hot(label, num_classes=3)
        output = output.squeeze()
        loss = criterion(output, label)

        err_ms = accuracy(output.data, label)
        if batch_no % 4 == 0:
            print ('\n\tBatch: '+ (str(batch_no))+'/'+str(len(train_load)) + ' Acc_(train set): ' + str(float(err_ms))+' Cross ent loss: ' + str(loss.item())[:5], end='#')
        else:
            print('#', end='')
            

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate(val_loader, model, criterion,optimizer, is_cuda):
    model.eval()

    for (label, sample) in enumerate(val_loader):
        if is_cuda:
            inp = sample[0:2]
            label = sample[2].cuda()
        else:
            inp = sample[0:2]
            label = sample[2]
        del sample
        
        output = model(inp)
        output = output.squeeze()
        # label = label-1
        loss = criterion(output, label)
        conf_matrix = np.zeros(shape=(3,3))
        print(output.argmax(axis = 1))
        print(label)
        for i, ohot_pred in enumerate(output.data):
            pred = ohot_pred.argmax()
            conf_matrix[pred,label[i]] += 1

        print(conf_matrix)

        err_ms = accuracy(output.data, label)
        print ('Acc (VAL SET): ' + str(err_ms.data) + 'valid loss cross_ent: ' + str(loss.item())[:5])


class PadSequence:
    def __call__(self, batch):

        # Let's assume that each element in "batch" is a tuple (data, label).
        # Sort the batch in the descending order
        # reshaper = map(lambda x: x[0].view(x[0].shape[0], -1))
        #batch = list(, batch))
        is_cuda = True
        if os.getcwd() == '':
            is_cuda = False
        #print('Cuda is {}'.format(is_cuda))

        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        # Get each sequence and pad it
        # sequences = [x.view(x[0].shape[0], -1) for x in sorted_batch]
        sequences = [x[0][:,None] for x in sorted_batch]
        dummy_seq = torch.zeros((526,*sequences[0].shape[1:]))
        sequences.insert(0, dummy_seq)

        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        sequences_padded = sequences_padded[1: , ...]

        # Also need to store the length of each sequence
        # This is later needed in order to unpad the sequences
        lengths = torch.LongTensor([len(x) for x in sequences[1:]])

        # Don't forget to grab the labels of the *sorted* batch
        labels = torch.LongTensor(list(map(lambda x: x[1], sorted_batch)))
        if is_cuda:
            sequences_padded = sequences_padded.cuda()

        # return sequences[1:], lengths, labels
        return sequences_padded, lengths, labels

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred    = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    total_acc = correct.sum().float()/len(target)

    # class_0_acc = []
    # class_1_acc =
    # class_2_acc =

    return total_acc

if __name__ == "__main__":
    init_train(30)