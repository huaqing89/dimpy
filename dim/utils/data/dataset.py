class Dataset():
  def __init__(self, data,feature=None):
    self.len = data.shape[0]
    if (feature is not None):
      self.data = data[:,:feature]
      self.label  = data[:,feature:]
    else:
      feature = data.shape[1]
      self.data = data[:,:feature]
      self.label = None 
  def __getitem__(self, index):
    if (self.label is None):
      return self.data[index]
    return self.data[index],self.label[index]
  def __len__(self):
    return self.len
