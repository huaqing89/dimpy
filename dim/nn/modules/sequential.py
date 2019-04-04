import dim
from .module import Module

class Sequential(Module):
  def __init__(self,*modules):
    super(Sequential,self).__init__()
    for i,x in enumerate(modules):
      self.moduleList.append({"name":str(i),"module":x})
    self.count=len(self.moduleList)
  
  def forward(self,x):
    for a in self.moduleList:
      x=a["module"].forward(x)
    return x
  
  def __str__(self):
    return "Sequential(\n"\
           +"\n".join(list("    ({}):{}".format(x["name"],x["module"]) for x in self.moduleList))\
           +"\n)"        
