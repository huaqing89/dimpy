import dim
from .module import Module

class ReLU(Module):
  def __init__(self):
    super(ReLU,self).__init__()
    self.moduleList.append({"name":None,"module":self}) 
    
  def forward(self,x):
    return dim.nn.functional.relu(x)
  def __str__(self):
    return "ReLu()"
