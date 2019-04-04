import math
import dim
from .module import Module

class Linear(Module):
  def __init__(self,inF,outF,bias=True):
    super(Linear,self).__init__()
    self.ins = inF
    self.outs = outF
    self.params=[]

    self.weight=self.setParameters((self.ins,self.outs))
    self.weight.setGrad()
    self.params.append(self.weight)

    if (bias):
      self.bias=self.setParameters((self.outs,))
      self.bias.setGrad()
      self.params.append(self.bias)
    else:
      self.bias=None
    
    self.moduleList.append({"name":None,"module":self}) 
  def setParameters(self,shape):
    stdv = 1. / math.sqrt(shape[-1])
    return dim.uniform(-stdv,stdv,shape)
    
  def forward(self,inputs):
    if (inputs.shape[1]!=self.ins or inputs.ndim!=2): raise Exception("参数[{}]不符合要求{},{}".format(inputs.shape,self.ins,inputs.ndim))
    if (self.bias is not None): return inputs.dot(self.weight).add(self.bias)
    return inputs.dot(self.weight)    
  
  def __str__(self):
    return "Linear(in_features={}, out_features={}, bias={})".format(self.ins,self.outs,self.bias is not None)
