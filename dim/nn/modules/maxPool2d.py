import dim
from .module import Module

class MaxPool2d(Module):
  def __init__(self,ks,padding):
    super(MaxPool2d,self).__init__()
    self.ks= ks
    self.indices = []
    self.padding = padding
    self.params=[]
    self.moduleList.append({"name":None,"module":self}) 
  
  def forward(self,x):
    self.X=x
    self.result = dim.nn.functional.maxPool2d(x,self.ks,self.indices,self.padding)
    return self.result
  
  def __str__(self):
    return "MaxPool2d(kernelSize={}, padding={})".format(self.ks,self.padding)
