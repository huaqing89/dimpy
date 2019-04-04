import dim
from .module import Module

class CrossEntropyLoss(Module):
  def __init__(self):
    super(CrossEntropyLoss,self).__init__()
    self.moduleList.append({"name":None,"module":self}) 
  def forward(self,x,y):
    return dim.nn.functional.crossEntropy(x,y)
  def __str__(self):
    return "CrossEntropyLoss()"
