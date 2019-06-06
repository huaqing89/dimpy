import json
import math

import dim
from .autograd import Autograd
from .constant import Constant

class Variable(Autograd):
  sequence=0
  def __init__(self,data,name=None):
    super(Variable,self).__init__()
    Variable.sequence+=1
    if (not name): name = "var"+str(Variable.sequence)
    self.name  = name
    self.data = data
    self.type = "Variable"
  def partGrad(self,partial={},prevOp=None):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (prevOp is None): 
      data=self.eval()
      if isinstance(data,dim.Vector):
        prevOp=Constant(dim.ones(data.shape))
      elif isinstance(data,dim.cl.Array):
        prevOp=Constant(dim.cl.ones(data.shape))
    #if (prevOp is None): prevOp=Constant(1)
    
    if (partial.name == self.name):
      rst = prevOp
      '''if (self.isNumber(self.data)): #标量
        if(partial.data.ndim==1):# 
          #console.log("标量对标量求导")
          rst= Constant(dim.ones(partial.data.size)*prevOp)
        elif (partial.data.ndim==2):
          #console.log("标量对矩阵求导")
          rst= Constant(dim.ones(partial.data.shape).T*prevOp)
        else: 
          rst = Constant(prevOp)
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
      '''
    else:
      #console.log("对非自身变量求导为0")
      rst = Constant(0)
    self._grads[self.name]=rst
    return rst

  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "{}[{}]".format(self.name,"*".join(str(i) for i in self.data.shape))
    self._expressionStr = rst
    return rst
  def eval(self,useCatch=True): return self.data.view()
  def backward(self,prevOp,partial):
    return self.partGrad(self)
  def variables(self): return [self]
  def isSame(self,a):
    if (not (isinstance(a,Variable))): return False
    if (self.name == a.name): return True
    return False
