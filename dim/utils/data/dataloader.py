import dim
from .dataset import Dataset
class _DataLoader():
  def __init__(self,obj):
    self.dataset = obj.dataset
    self.shuffle = obj.shuffle
    self.batchSize = obj.batchSize
    self.dropLast = obj.dropLast
    
    self.indicesIter = iter(obj.indices)
  def __iter__(self):
    return self
  def __next__(self):
    return self.dataset[next(self.indicesIter)]
class DataLoader():
  def __init__(self,dataset=None,batchSize=1,shuffle=False,dropLast=False):
    if not isinstance(dataset,Dataset):
      raise Exception("dataset must be Dataset type")
    self.batchSize=batchSize
    self.shuffle = shuffle
    self.dataset = dataset
    self.dropLast = dropLast
    self.batches=0
    if (not shuffle):
      self.indices = Indices(list(range(len(self.dataset))),batchSize,dropLast)
    else:
      self.indices = Indices(dim.shuffle(list(range(len(self.dataset)))),batchSize,dropLast)

  def __iter__(self):
    return _DataLoader(self)

class Indices():
  def __init__(self, indices, batchSize, dropLast):
    self.indices = indices
    self.batchSize = batchSize
    self.dropLast = dropLast

  def __iter__(self):
    batch = []
    for idx in self.indices:
        batch.append(int(idx))
        if len(batch) == self.batchSize:
            yield batch
            batch = []
    if len(batch) > 0 and not self.dropLast:
        yield batch

  def __len__(self):
    if self.dropLast:
        return len(self.indices) // self.batchSize
    else:
        return (len(self.indices) + self.batchSize - 1) // self.batchSize
