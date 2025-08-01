import os
import time
import torch
import argparse
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from datahelper.dataset import DTADataset
from models.model import MutualDTA
from utils import *


# training function at each epoch
def train(model,  train_loader, optimizer, epoch, writer = None):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    train_loss = 0
    model.train()
    start_time = time.time()
    for batch_idx, data in enumerate(train_loader):
        compounds, proteins, Y = data
        optimizer.zero_grad()
        output = model(compounds,proteins)
        loss = F.mse_loss(output, Y.reshape(-1,1).float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if batch_idx % 20 == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(epoch,
                                                                           batch_idx * len(Y),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
    train_loss /= len(train_loader)
    print('Train epoch: {} \tLoss: {:.4f}'.format(epoch, train_loss),'Spend Time',time.time()-start_time)

    if writer:
        writer.add_scalar('Loss/Train', train_loss, epoch+1)

    return train_loss

def predicting(model, loader, epoch):
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    model.eval()
    test_loss = 0
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        for data in loader:
            compounds, proteins, Y = data
            output = model(compounds, proteins)
            loss = F.mse_loss(output, Y.reshape(-1,1).float())
            test_loss += loss.item()
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, Y.float().cpu()), 0)
    test_loss /= len(loader)
    print('Test\Valid epoch:{}\tLoss:{:.4f}'.format(epoch, test_loss))
    return test_loss, total_labels.numpy().flatten(), total_preds.numpy().flatten()

# eval
def eval(result_file_path, epoch, test_mse, model_st, dataset, y_true, y_pred, writer=None):
    mse_value = mse(y_true, y_pred)
    rm2 = get_rm2(y_true, y_pred)
    ci_value = get_cindex(y_true, y_pred)

    if writer:
        writer.add_scalar('Loss/Test', test_mse, epoch + 1)
        writer.add_scalar('rm2', rm2, epoch + 1)
        writer.add_scalar('ci', ci_value, epoch + 1)

    if not os.path.exists(result_file_path):
        with open(result_file_path, 'w') as f:
            f.write('\nepoch,test_mse,test_rm_2,test_ci\n')

    with open(result_file_path, 'a+') as f:
        f.write(','.join(map(str, [epoch, mse_value, rm2, ci_value])) + '\n')

    print('epoch ', epoch, 'test_mse,test_ci,test_rm_2:', mse_value, ci_value, rm2, model_st, dataset)

    return mse_value, ci_value, rm2

def main(args):
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter()
    except Exception:
        writer = None

    set_seed(42)
    dataset = args.dataset
    device = args.device

    model_map = {
        'MutualDTA':MutualDTA,
    }
    modeling = model_map[args.model]
    model_st = modeling.__name__
    print('Learning rate: ', args.lr)
    print('Epochs: ', args.max_epoch)
    print('\nrunning on ', model_st + '_' + dataset)
    print('use',device)


    train_data = DTADataset(args.data_path, dataset, 'train', args.cold_start, device)
    test_data = DTADataset(args.data_path, dataset, 'test', args.cold_start, device)
    # make data PyTorch mini-batch processing ready
    train_loader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True, collate_fn=train_data.collate_fn)
    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, collate_fn=test_data.collate_fn)

    model = modeling(args.n_durg_features, args.n_protein_features, args.n_edge_features, args.protein_hidden_dim).to(device)
    os.makedirs(args.save_path,exist_ok=True)
    model_file_path = os.path.join(args.save_path,'model_' + model_st + '_' + dataset + '.model')
    result_file_path = os.path.join(args.save_path,'result_' + model_st + '_' + dataset + '.csv')
    
    optimizer = Adam(model.parameters(),lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer,10,eta_min=0.0001,verbose=True)

    init_epoch = 0
    best_mse = 1000
    best_ci = 0
    best_rm_2 = 0
    best_epoch = -1

    if args.init_model or args.test_only:
        print('Loading checkpoint ...')
        checkpoint = torch.load(model_file_path)
        init_epoch = checkpoint['epoch']
        print('epoch:',init_epoch)
        print('model_state_dict','\n'.join(k+' '+str(v.shape) for k,v in checkpoint['model_state_dict'].items()))
        model.load_state_dict(checkpoint['model_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # scheduler.last_epoch = init_epoch
        test_loss, G, P = predicting(model, test_loader, init_epoch + 1)
        eval(result_file_path, init_epoch, test_loss, model_st, dataset, G, P, writer)

    # training the model
    for epoch in range(init_epoch,args.max_epoch):

        train(model, train_loader, optimizer, epoch + 1, writer)
        
        if epoch % args.test_interval == 0:
            test_loss, G, P = predicting(model, test_loader, epoch + 1)
            test_loss, ci_value, rm2 = eval(result_file_path, epoch, test_loss, model_st, dataset, G, P, writer)
            scheduler.step()

            if test_loss < best_mse or ci_value > best_ci:
                best_epoch, best_mse, best_ci, best_rm_2 = epoch, test_loss, ci_value, rm2
                if epoch:
                    torch.save({'model_state_dict': model.state_dict(),'epoch': epoch,'optimizer_state_dict':optimizer.state_dict(),'scheduler_state_dict':scheduler.state_dict()}, model_file_path)
                print('mse or ci improved at epoch ', best_epoch, '; best_mse,best_ci,best_rm_2:', best_mse,best_ci,rm2,model_st,dataset)
            else:
                print('No improvement since epoch ', best_epoch, '; best_mse,best_ci,best_rm_2:', best_mse,best_ci,best_rm_2,model_st,dataset)

    print('dataset',dataset,'best epoch ', best_epoch, '; best_mse,best_ci,best_rm_2:', best_mse,best_ci,best_rm_2,model_st,dataset)
    print(args)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset', type=str, default='davis')
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=128)
    parser.add_argument("--max_epoch", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=0.0008)
    parser.add_argument("--data_path", type=str,default='./data')
    parser.add_argument("--save_path", type=str,default='./experiments/train')
    
    parser.add_argument('-init', '--init_model', action='store_true')
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--test_interval', default=1, type=int)

    parser.add_argument("--model", default='MutualDTA', type=str)
    parser.add_argument("--cold_start", default='',help='drug or protein or None', type=str)
    parser.add_argument("--n_durg_features", type=int, default=512, help='number of drug features')
    parser.add_argument("--n_protein_features", type=int, default=256, help='number of protein features')
    parser.add_argument("--n_edge_features", type=int, default=6, help='number of edge features')
    parser.add_argument("--protein_hidden_dim", type=int, default=256, help='hidden dimension for protein features')

    args = parser.parse_args()
    main(args)

