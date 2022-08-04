from BaselineNet import Main_Net
import torch.optim as optim
from train import Trainer
from config import Config
from data_loader1 import DatasetVal
from torch.utils.data import DataLoader
import torch
import os
from apex import amp

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    torch.cuda.set_device(0)

    val_dataset = DatasetVal()
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0)

    net = Main_Net()
    net = net.cuda()

    cfg = Config()
    trainer = Trainer(net, cfg, './log')
    trainer.load_weights('./model_codnet.pt')
    trainer.eval(val_loader, val_dataset)
