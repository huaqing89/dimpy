from torch.utils.data import Dataset,DataLoader
batch_size=100
x=torch.randint(0,10,(1000,4))
y=(x[:,0]*x[:,1]+x[:,2]*x[:,3]).reshape(1000,1)
train_data ,test_data = torch.cat((x,y),1).split(int(x.size(0)*0.8))
print(train_data.shape)
class MyDataset(Dataset):
    def __init__(self, data,feature):
        self.len = data.shape[0]
        self.data = data[:,:feature]
        self.label  = data[:,feature:]
    def __getitem__(self, index):
        return self.data[index],self.label[index]
    def __len__(self):
        return self.len

train_loader = DataLoader(dataset=MyDataset(train_data,4),batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=MyDataset(test_data,4),batch_size=batch_size, shuffle=True)
print("train...")
for data,label in train_loader:
    print(data.size(),label.size(),data.sum())
print("test...")    
for data,label in test_loader:
    print(data.size(),label.size(),data.sum())