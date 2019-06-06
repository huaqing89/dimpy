import torch
from torch import nn,otim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets,transforms

batch_size=64
lr=1e-2
num_epoches=20

data_tf = transforms.Compose(
  [transforms.ToTensor(),
   transforms.Normalize([0.5],[0.5])
  ])
train_dataset = datasets.MNIST(
  root='./data',train=True,transform=data_tf,download=True)

test_dataset = datasets.MNIST(
  root='./data',train=False,transform=data_tf,download=True)

train_loader=DataLoader(train_dataset,batch_size,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size,shuffle=False)

class Net(nn.Module):
  def __init__(self,in_dim,n_hidden1,n_hidden2,out_dim):
    super(Net,self).__init__()
    self.seq = nn.Sequential(
      nn.Linear(in_dim,n_hidden1),
      nn.ReLU(),
      nn.Linear(n_hidden1,n_hidden2),
      nn.ReLU(),
      nn.Linear(n_hidden2,out_dim)
    )
  def forward(self,x):
    x=self.seq(x)
    return x 
    
model=new.simpleNet(28*28,300,100,10)
criterion = nn.CrossEntropyLoss()
optimer = optim.SGD(model.parameters(),lr=lr)

model.eval()
print(model)
eval_loss=0
eval_acc=0
for data in test_loader:
    img,label = data
    img = img.view(img.size(0),-1)
print(data)
#print(img[0])
print(label[0:11])
out=model(img)
loss=criterion(out,label)
eval_loss += loss.data[0]*label.size(0)
_,pred=torch.max(out,1)
num_correct=(pred==label).sum()
eval_acc += num_correct.data[0]
print('Test Loss:{:.6f},Acc:{:.6f}'.format(eval_loss/(len(test_dataset)),eval_acc / (len(test_dataset))))

