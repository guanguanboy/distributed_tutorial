import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2" #设置本脚本可见的GPU，如果不设置，则全部都可见

input_size = 5
output_size = 2
batch_size = 30
data_size = 3000

class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len



class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("  In Model: input size", input.size(),
              "output size", output.size())
        return output

def main():

    rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                            batch_size=batch_size, shuffle=True)
                            
    model = Model(input_size, output_size)

    if torch.cuda.is_available():
        model.cuda()
        
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # 就这一行
        model = nn.DataParallel(model) #只需这一行就可以实现多GPU并行训练
        #model = nn.DataParallel(model, device_ids=[0, 1, 2]) #也可以通过这种方式来指定所使用的GPU
        
    for data in rand_loader:
        if torch.cuda.is_available():
            input_var = Variable(data.cuda())
        else:
            input_var = Variable(data)
        output = model(input_var)
        print("Outside: input size", input_var.size(), "output_size", output.size())

if __name__=="__main__":
    main()
