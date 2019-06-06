import sys
sys.path.append("..")
import time
import dim
from torch.utils.data import Dataset,DataLoader
x=dim.rand(1000,5)
y=dim.randint(0,9,(1000,1))
seq=dim.nn.Sequential(\
  dim.nn.Linear(5,10),
  dim.nn.ReLU(),
  dim.nn.Linear(10,5),
  dim.nn.ReLU(),
  dim.nn.Linear(5,1)
)
#g=dim.nn.functional.crossEntropy(m2,y)

class MyDataset(Dataset):

    def __init__(self, data):
        self.len = data.shape[0]
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

loader = DataLoader(dataset=MyDataset(x),batch_size=batch_size, shuffle=True)

start=time.time()
optim = dim.optim.Adam(seq.parameters(),{"lr":0.0001})
criterion = dim.nn.MSELoss()
pred=seq.forward(x)
loss=criterion.forward(pred,y)
for i in range(50000):
  loss.backward()
  optim.step()
  optim.zeroGrad()
  loss.gradFn.clearData()
  if (i%5000==0):
    print("time=",time.time(),"epoch=",i,"loss=",loss.gradFn.eval().value())
end=time.time()
print("total:",end-start)