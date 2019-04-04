import dim
from .module import Module

class Conv2d(Module):
  def __init__(self,inChannels,outChannels,kernelSize,stride=1,padding=0,bias=False):
    super(Conv2d,self).__init__()
    self.inChannels = inChannels
    self.outChannels = outChannels
    self.kernelSize = kernelSize
    self.stride=stride
    self.padding = padding
    self.bias = bias
    self.params=[]
    
    self.weight=((self.outChannels,self.inChannels,self.kernelSize,self.kernelSize))
    self.weight.setGrad()
    self.params.append(self.weight)
    
    if (bias):
      self.bias=self.setParameters((inputs.shape[0],self.outChannels,self.kernelSize))
      self.bias.setGrad()
      self.params.append(self.bias)
    
    self.moduleList.append({"name":None,"module":self}) 
  
  def setParameters(self,shape):
    stdv = 1. / math.sqrt(shape[-1])
    return dim.uniform(-stdv,stdv,shape)
    
  def forward(self,inputs):
    self.Input=inputs
    
  
    if (self.bias is not None): return dim.nn.functional.conv2d(self.Input,self.Filter,self.stride,self.padding).add(self.bias)
    return dim.nn.functional.conv2d(self.Input,self.Filter,self.stride,self.padding)
  
  def __str__(self):
    return "Conv2d(inChannels={}, outChannels={}, kernelSize={},stride={},padding={},bias={})".format(self.inChannels,self.outChannels,self.kernelSize,self.stride,self.padding,self.bias is not None)
