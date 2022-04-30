#training

from .dglGIN import *
from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info
from dgl.dataloading import GraphDataLoader
import os
import sys
import argparse


def parse_arguments(parser):
    #parser.add_argument('--device', type=str,choices=['cpu','gpu'], default='cpu')
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--data_path', type=str, default='../data/graph.bin')
    parser.add_argument('--save_path', type=str, default='../QAsave/')
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--num_mlp_layers', type=int, default=2)
    parser.add_argument('--input_dim', type=int, default=29)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--output_dim', type=int, default=2)
    parser.add_argument('--final_dropout', type=float, default=0.5)
    parser.add_argument('--learn_eps', type=bool, default=1)
    parser.add_argument('--graph_pooling_type', type=str,choices=['sum','mean','max'], default='sum')
    parser.add_argument('--neighbor_pooling_type', type=str, choices=['sum','mean','max'],default='sum')
    args = parser.parse_args()
    return args
    

class QAdataset(DGLDataset):
    def __init__(self):
        super(QAdataset, self).__init__(name='QAdataset')
    def process(self,graph_path):
        self.graphs, label_dict = load_graphs(graph_path)
        self.labels = label_dict['labels']
    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]    
    def __len__(self):
        return len(self.graphs)


def train(net, trainloader, optimizer, criterion, epoch):
    net.train()

    running_loss = 0
    total_iters = len(trainloader)
    batch_acc = []
    batch_loss = []
    batch_num = []

    for graphs, labels in trainloader:
        labels = labels.to('cpu')
        graphs = graphs.to('cpu')
        feat = torch.cat((graphs.ndata['AtomSymbol'],graphs.ndata['AtomInRing'],graphs.ndata['AtomHybridization'],graphs.ndata['ImplicitValence'],graphs.ndata['AtomDegree']),1)
        outputs = net(graphs, feat)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels.data).sum().item()
        batch_acc.append(correct/len(labels))
        loss = criterion(outputs, labels)
        batch_loss.append(loss.item())
        running_loss += loss.item()
        batch_num.append(len(labels))
        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    return batch_acc,batch_loss,batch_num
    
def eval_net(net, dataloader, criterion):
    net.eval()

    total = 0
    total_loss = 0
    total_correct = 0

    for data in dataloader:
        graphs, labels = data
        graphs = graphs.to('cpu')
        labels = labels.to('cpu')
        feat = torch.cat((graphs.ndata['AtomSymbol'],graphs.ndata['AtomInRing'],graphs.ndata['AtomHybridization'],graphs.ndata['ImplicitValence'],graphs.ndata['AtomDegree']),1)
        total += len(labels)
        outputs = net(graphs, feat)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels.data).sum().item()
        total_correct += correct
        loss = criterion(outputs, labels)
        # crossentropy(reduce=True) for default
        total_loss += loss.item() * len(labels)

    loss, acc = 1.0*total_loss / total, 1.0*total_correct / total

    net.train()

    return loss, acc

if __name__ == "__main__":

    torch.manual_seed(seed=1234)
    np.random.seed(seed=1234)
    
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    
    class QAdataset(DGLDataset):
        def __init__(self, raw_dir=None):
            super(QAdataset, self).__init__(name='QAdataset',raw_dir=raw_dir)
        def process(self):
            graph_path = args.data_path
            self.graphs, label_dict = load_graphs(graph_path)
            self.labels = label_dict['labels']
        def __getitem__(self, idx):
            return self.graphs[idx], self.labels[idx]    
        def __len__(self):
            return len(self.graphs)

    dataset = QAdataset()
    
    trainloader, validloader = GINDataLoader(
        dataset, batch_size=args.batch_size, device=torch.device('cpu'),
        seed=1234, shuffle=True,
        split_name='fold10', fold_idx=0).train_valid_loader()
        
    model = GIN(
        args.num_layers,args.num_mlp_layers,args.input_dim,args.hidden_dim,args.output_dim,args.final_dropout,args.learn_eps,args.graph_pooling_type,args.neighbor_pooling_type).to('cpu')    
        
    criterion = nn.CrossEntropyLoss()  # defaul reduce is true
    optimizer = optim.Adam(model.parameters(), args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
            
    for epoch in range(args.epoch):

        batch_acc, batch_loss, batch_num=train(model, trainloader, optimizer, criterion, epoch)
        scheduler.step()

        valid_loss, valid_acc = eval_net(
            model, validloader, criterion)

        with open(args.save_path+'GIN_loss.txt', 'a') as f:
            f.write('epoch'+str(epoch+1)+' ')
            f.write("%f %f" % (
                valid_loss,
                valid_acc
            ))
            f.write("\n")

        with open(args.save_path+'GIN_loss_batch.txt','a') as f:
            f.write('epoch'+str(epoch+1)+' \n')
            for i in range(len(batch_acc)):
                f.write("%f %f %f"% (batch_acc[i],batch_loss[i],batch_num[i]))
                f.write('\n')

    torch.save(model.state_dict(), args.save_path+'model_QA_GIN.pt')