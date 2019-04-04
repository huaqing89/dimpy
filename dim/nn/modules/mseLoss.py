import dim
from .module import Module

class MSELoss(Module):
  def __init__(self):
    super(MSELoss,self).__init__()
    self.moduleList.append({"name":None,"module":self}) 
    
  def forward(self,x,y):
    return dim.nn.functional.mseLoss(x,y)
  def __str__(self):
    return "MSELoss()"
