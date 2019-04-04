import json
import math
import random
import hashlib

import dim
from .autograd import Autograd
from .constant import Constant
VARIABLE=[]

class Variable(Autograd):
  def __init__(self,data,name=None):
    super(Variable,self).__init__()
    if (not name): name = "var"+str(random.random())[-6:]
    self.name  = name
    self.data = data
    self.type = "Variable"
    VARIABLE.append({"name":self.name})  
  def partGrad(self,partial={},prevOp=None):
    if (prevOp is None): prevOp=Constant(1)
    if (partial.name == self.name):
      return prevOp
    else:
      return Constant(0)
    if (partial.name == self.name):
      if (self.isNumber(self.data)): #标量
        if(partial.data.ndim==1):# 
          #console.log("标量对标量求导")
          rst= Constant(dim.ones(partial.data.size))
        elif (partial.data.ndim==2):
          #console.log("标量对矩阵求导")
          rst= Constant(dim.ones(partial.data.shape).T)
        else: 
          rst = Constant(1)
      elif (self.data.ndim==1): #向量
        if self.isNumber(partial.data):
          #console.log("向量对标量求导")
          pass
        elif (partial.data.ndim==1):
          #console.log("向量对向量求导，理论应该是返回雅可比矩阵")
          rst = Constant(dim.ones(self.data.size))
        else:
          raise Exception("不支持向量关于矩阵的求导运算")
      elif (self.data.ndim==2): #矩阵
        if self.isNumber(partial.data):
          #console.log("矩阵对标量求导")
          rst = Constant(dim.ones(self.data.shape))
        else: 
          #console.log("矩阵对矩阵求导")
          rst = Constant(dim.ones(self.data.shape))
          #raise Exception("不支持矩阵关于向量或矩阵的求导运算")
      else:
        raise Exception("不支持超过两维的高阶求导")
    else:
      #console.log("对非自身变量求导为0")
      rst = Constant(0)
    self._grads[self.name]=rst
    return rst

  def expression(self):
    return self.name
  def eval(self): return self.data
  def backward(self): return self.partGrad(self)
  def variables(self): return [self]
  def isSame(self,a):
    if (not (isinstance(a,Variable))): return False
    if (self.name == a.name): return True
    return False
