import json
import math
import random
import hashlib

import dim
from .autograd import Autograd
from .constant import Constant

OPERATE=[]

class Operate(Autograd):
  def __init__(self,left,right,operate,args=None,name=None):
    super(Operate,self).__init__()
    if self.isNumber(left): self.left = Constant(left)
    else: self.left = left
    if self.isNumber(right): self.right = Constant(right)
    else: self.right = right
    self.args = args
    if (not name): name = "op"+str(random.random())[-6:]
    self.name = name 
    
    self.operate = operate
    self.type = "Operate"
  
  def partGrad(self,partial,prevOp):pass
  def expression(self):pass
  def eval(self):pass

  def variables(self,v=[]):
    if (self.left and self.left.type=="Operate"): v=self.left.variables(v)
    if (self.right and self.right.type=="Operate"): v=self.right.variables(v)
    if (self.left and self.left.type=="Variable"):
      try:
        list((x.name for x in v)).index(self.left.name)
      except:
        v.append(self.left)
    if (self.right and self.right.type=="Variable"):
      try:
        list((x.name for x in v)).index(self.right.name)
      except:
        v.append(self.right) 
    return v

  @staticmethod 
  def wrapper(left,right,operate,args=None,name=None):
    if (operate=="add"): return dim.autograd.AddOperate(left,right,args,name)
    if (operate=="sub"): return dim.autograd.SubOperate(left,right,args,name)
    if (operate=="mul"): return dim.autograd.MulOperate(left,right,args,name)
    if (operate=="div"): return dim.autograd.DivOperate(left,right,args,name)
    if (operate=="pow"): return dim.autograd.PowOperate(left,right,args,name)
    if (operate=="exp"): return dim.autograd.ExpOperate(left,right,args,name)
    if (operate=="log"): return dim.autograd.LogOperate(left,right,args,name)
    if (operate=="log2"): return dim.autograd.Log2Operate(left,right,args,name)
    if (operate=="log10"): return dim.autograd.Log10Operate(left,right,args,name)
    if (operate=="sin"): return dim.autograd.SinOperate(left,right,args,name)
    if (operate=="cos"): return dim.autograd.CosOperate(left,right,args,name)
    if (operate=="tan"): return dim.autograd.TanOperate(left,right,args,name)
    if (operate=="asin"): return dim.autograd.AsinOperate(left,right,args,name)
    if (operate=="acos"): return dim.autograd.AcosOperate(left,right,args,name)
    if (operate=="atan"): return dim.autograd.Atanperate(left,right,args,name)
    if (operate=="sinh"): return dim.autograd.SinhOperate(left,right,args,name)
    if (operate=="cosh"): return dim.autograd.CoshOperate(left,right,args,name)
    if (operate=="tanh"): return dim.autograd.TanhOperate(left,right,args,name)
    if (operate=="asinh"): return dim.autograd.AsinhOperate(left,right,args,name)
    if (operate=="acosh"): return dim.autograd.AcoshOperate(left,right,args,name)
    if (operate=="atanh"): return dim.autograd.AtanhOperate(left,right,args,name)
    if (operate=="sum"): return dim.autograd.SumOperate(left,right,args,name)
    if (operate=="mean"): return dim.autograd.MeanOperate(left,right,args,name)
    if (operate=="max"): return dim.autograd.MaxOperate(left,right,args,name)
    if (operate=="min"): return dim.autograd.MinOperate(left,right,args,name)
    if (operate=="abs"): return dim.autograd.AbsOperate(left,right,args,name)

    if (operate=="dot"): return dim.autograd.DotOperate(left,right,args,name)
    if (operate=="T"): return dim.autograd.TOperate(left,right,args,name)
  
    if (operate=="relu"): return dim.autograd.ReluOperate(left,right,args,name)
    if (operate=="reluDeri"): return dim.autograd.ReluDeriOperate(left,right,args,name)
    if (operate=="sigmoid"): return dim.autograd.SigmoidOperate(left,right,args,name)
    if (operate=="sigmoidDeri"): return dim.autograd.SigmoidDeriOperate(left,right,args,name)
    if (operate=="softmax"): return dim.autograd.SoftmaxOperate(left,right,args,name)
    if (operate=="softmaxDeri"): return dim.autograd.SoftmaxDeriOperate(left,right,args,name)
    if (operate=="crossEntropy"): return dim.autograd.CrossEntropyOperate(left,right,args,name)
    if (operate=="crossEntropyDeri"): return dim.autograd.CrossEntropyDeriOperate(left,right,args,name)
    if (operate=="mseLoss"): return dim.autograd.MSELossOperate(left,right,args,name)
  
    if (operate=="conv1d"): return dim.autograd.Conv1dOperate(left,right,args,name)
    if (operate=="conv2d"): return dim.autograd.Conv2dOperate(left,right,args,name)
    if (operate=="convTranspose1d"): return dim.autograd.ConvTranspose1dOperate(left,right,args,name)
    if (operate=="convTranspose2d"): return dim.autograd.ConvTranspose2dOperate(left,right,args,name)
    if (operate=="maxPool1d"): return dim.autograd.MaxPool1dOperate(left,right,args,name)
    if (operate=="avgPool1d"): return dim.autograd.AvgPool1dOperate(left,right,args,name)
    if (operate=="maxPool2d"): return dim.autograd.MaxPool2dOperate(left,right,args,name)
    if (operate=="avgPool2d"): return dim.autograd.AvgPool2dOperate(left,right,args,name)
    if (operate=="maxUnpool1d"): return dim.autograd.MaxUnpool1dOperate(left,right,args,name)
    if (operate=="avgUnpool1d"): return dim.autograd.AvgUnpool1dOperate(left,right,args,name)
    if (operate=="maxUnpool2d"): return dim.autograd.MaxUnpool2dOperate(left,right,args,name)
    if (operate=="avgUnpool1d"): return dim.autograd.AvgUnpool2dOperate(left,right,args,name)

    raise Exception("未定义的操作")

  def backward(self,prevOp,partial):
    if (not partial): partial=self.variables()[0]
    if (not partial or partial.type!="Variable"): raise Exception('partial参数必须Variable类型')
    return self.partGrad(partial,prevOp)

  def isSame(self,a):
    if (a.type!="Operate"): return False
    leftEqual =  self.left.isSame(a.left)
    if (a.right is None and self.right is None):
      rightEqual = True
    elif (a.right is None or self.right is None):
      rightEqual = False
    else:
      rightEqual = self.right.isSame(a.right)
    
    if (leftEqual and rightEqual and self.type=="Operate"):
      if (self.operate == a.operate): return True
    return False

  def shrink(self):
    left = self.left.factor("add")
    right = self.right.factor("add")
    for i in left:
      for j in right:
        print("add:",i.name,"=",j.name)
        if (i.isSame(j)): print("shrink",i,j)

    left = self.left.factor("mul")
    right = self.right.factor("mul")
    for i in left:
      for j in right:
        print("mul:",i.name,"=",j.name)
        if (i.isSame(j)): print("shrink",i,j)
  
  def factor(self,opStr,aFactor):
    if (aFactor is None): aFactor=[]
    print("factor:",self.operate)
    if (self.operate!=opStr): return aFactor
    aFactor.append(self.left)
    aFactor.append(self.right)
    self.left.factor(opStr,aFactor)
    self.right.factor(opStr,aFactor)
    return aFactor
