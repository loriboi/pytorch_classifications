import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import json
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms
import pandas as pd
import torch.nn.functional as F



class HyperParameters():
    def __init__(self,model,num_epochs,criterion,optimizer,lr,weight_decay):
        self.setDevice()
        self.num_epochs = num_epochs
        self.model = model
        if criterion == "CEL":
            self.criterion = nn.CrossEntropyLoss()

        if optimizer == "Adam" and weight_decay:
            self.optimizer = optim.Adam(self.model.parameters(), lr,weight_decay = 1e-5)
            self.optimizerInfo = { "OptName":"Adam", "lr":str(lr), "weight_decay": "True"}
        elif optimizer == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(),lr)
            self.optimizerInfo = { "OptName":"Adam", "lr":str(lr), "weight_decay": "False"}
        elif optimizer == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr, momentum=0.9, weight_decay=1e-5)
            self.optimizerInfo = { "OptName":"SGD", "lr":str(lr),"momentum":"0.9" ,"weight_decay": "True"}

        self.setSchedule()

    def setSchedule(self,scheduler = None, activate = False):
        self.schedulerON = activate
        if scheduler == "RLROnP":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3)
            self.schedulerName = scheduler
        elif scheduler == "SLR":
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
            self.schedulerName = scheduler


    def setDevice(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setName(self, name):
        os.makedirs(name,exist_ok=True)
        self.name = name
        self.checkpoint_path = os.path.join(name, name+".pth")
        self.latimeline = os.path.join(name,name+".txt")
    
    def saveModelInfo(self):
        self.info = {
            "name":self.name,
            "optimizer":self.optimizerInfo,
            "scheduler":self.schedulerName
        }
        path = os.path.join(self.name, self.name+"_info.txt")
        with open(path, 'w') as file:
            json.dump(self.info, file, indent=4)

    def shutdownatend(self,flag = False):
        if flag:
            os.system("shutdown /s /t 1")
        else:
            print("Train completed")

    def loadmodelcheckpoint(self):
        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    
    