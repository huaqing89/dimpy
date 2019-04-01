#coding:utf-8

from dim import Dim
from dim import Vector
dim = Dim()
import json
import math
import random
import hashlib

CONSTANT= []
VARIABLE=[]
OPERATE=[]

class Autograd(object):
  def __init__(self):
    super(Autograd,self).__init__()
    self._expressionStr=None
    self._grads={}
    self._data=None
    self.left=None
    self.right=None
    self.catch=True
    
  def setCatch(self,bool=True):
    if (bool):
      self.catch=True
    else:
      self.catch=False
      self._data=None
      self._grads={}
      self._expressionStr=None
    if (self.left): self.left.setCatch(bool)
    if (self.right): self.right.setCatch(bool)
    
  def clearData(self):
    self._data=None
    if (self.left): self.left.clearData()
    if (self.right): self.right.clearData()
  
  def findOp(self,name):
    if (self.type=='Operate' and self.name==name): return self
    left = self.left and self.left.findOp(name)
    right = self.right and self.right.findOp(name)
    if (left):
      return left
    elif (right):
      return right
    else:
      return {}
  def isNumber(sefl,val):
    return isinstance(val,int) or isinstance(val,float)
  def shrink(self): pass
  
  def factor(self,opStr): return [self]
  def gradExpression(self):
    m=[]
    for x in self._grads:
      m.append({"name":x,"expression":self._grads[x].expression()})
    return m

class Constant(Autograd):
  def __init__(self,data):
    super(Constant,self).__init__()
    if isinstance(data,Constant): return self
    if self.isNumber(data): data=dim.vector(data)
    md5=hashlib.md5()
    md5.update(json.dumps(data.tolist()).encode())
    hashData = md5.hexdigest()  
    try:
      idx=list((x["hash"] for x in CONSTANT)).index(hashData)
      obj = CONSTANT[idx]["object"]
      self.name=obj.name
      self.data=obj.data
    except :
      self.name = "const"+str(random.random())[-6:]
      self.data = data
      CONSTANT.append({"name":self.name,"hash":hashData,"object":self})
    self.type = "Constant"
    self._expressionStr=self.name
  
  def partGrad(self,partial,prevOp):
    rst = Constant(0)
    self._grads[self.name]=rst
    return rst

  def expression(self):
    if self.isNumber(self.data): return str(self.data)
    rst=json.dumps(self.data.tolist())
    if (len(rst)>50): return "Data[{}]".format("*".join(str(i) for i in self.data.shape)) 
    else: return rst

  def eval(self):return self.data
  def backward(self):return self.partGrad()
  def variables(self):return []
  def isSame(self,a):
    if (not isinstance(a,Constant)): return False
    if (self.name==a.name and self.data==a.data): return True
    return Fasle

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
    if (operate=="add"): return AddOperate(left,right,args,name)
    if (operate=="sub"): return SubOperate(left,right,args,name)
    if (operate=="mul"): return MulOperate(left,right,args,name)
    if (operate=="div"): return DivOperate(left,right,args,name)
    if (operate=="pow"): return PowOperate(left,right,args,name)
    if (operate=="square"): return SquareOperate(left,right,args,name)
    if (operate=="sqrt"): return SqrtOperate(left,right,args,name)
    if (operate=="exp"): return ExpOperate(left,right,args,name)
    if (operate=="log"): return LogOperate(left,right,args,name)
    if (operate=="log2"): return Log2Operate(left,right,args,name)
    if (operate=="log10"): return Log10Operate(left,right,args,name)
    if (operate=="sin"): return SinOperate(left,right,args,name)
    if (operate=="cos"): return CosOperate(left,right,args,name)
    if (operate=="tan"): return TanOperate(left,right,args,name)
    if (operate=="asin"): return AsinOperate(left,right,args,name)
    if (operate=="acos"): return AcosOperate(left,right,args,name)
    if (operate=="atan"): return Atanperate(left,right,args,name)
    if (operate=="sinh"): return SinhOperate(left,right,args,name)
    if (operate=="cosh"): return CoshOperate(left,right,args,name)
    if (operate=="tanh"): return TanhOperate(left,right,args,name)
    if (operate=="asinh"): return AsinhOperate(left,right,args,name)
    if (operate=="acosh"): return AcoshOperate(left,right,args,name)
    if (operate=="atanh"): return AtanhOperate(left,right,args,name)
    if (operate=="sum"): return SumOperate(left,right,args,name)
    if (operate=="mean"): return MeanOperate(left,right,args,name)
    if (operate=="max"): return MaxOperate(left,right,args,name)
    if (operate=="min"): return MinOperate(left,right,args,name)
    if (operate=="abs"): return AbsOperate(left,right,args,name)

    if (operate=="dot"): return DotOperate(left,right,args,name)
    if (operate=="T"): return TOperate(left,right,args,name)
  
    if (operate=="relu"): return ReluOperate(left,right,args,name)
    if (operate=="reluDeri"): return ReluDeriOperate(left,right,args,name)
    if (operate=="sigmoid"): return SigmoidOperate(left,right,args,name)
    if (operate=="sigmoidDeri"): return SigmoidDeriOperate(left,right,args,name)
    if (operate=="softmax"): return SoftmaxOperate(left,right,args,name)
    if (operate=="softmaxDeri"): return SoftmaxDeriOperate(left,right,args,name)
    if (operate=="crossEntropy"): return CrossEntropyOperate(left,right,args,name)
    if (operate=="crossEntropyDeri"): return CrossEntropyDeriOperate(left,right,args,name)
    if (operate=="mseLoss"): return MSELossOperate(left,right,args,name)
  
    if (operate=="conv1d"): return Conv1dOperate(left,right,args,name)
    if (operate=="conv2d"): return Conv2dOperate(left,right,args,name)
    if (operate=="transpose1d"): return Transpose1dOperate(left,right,args,name)
    if (operate=="transpose2d"): return Transpose2dOperate(left,right,args,name)
    if (operate=="maxPool1d"): return MaxPool1dOperate(left,right,args,name)
    if (operate=="avgPool1d"): return AvgPool1dOperate(left,right,args,name)
    if (operate=="maxPool2d"): return MaxPool2dOperate(left,right,args,name)
    if (operate=="avgPool2d"): return AvgPool2dOperate(left,right,args,name)
    if (operate=="maxUnpool1d"): return MaxUnpool1dOperate(left,right,args,name)
    if (operate=="avgUnpool1d"): return AvgUnpool1dOperate(left,right,args,name)
    if (operate=="maxUnpool2d"): return MaxUnpool2dOperate(left,right,args,name)
    if (operate=="avgUnpool1d"): return AvgUnpool2dOperate(left,right,args,name)

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

class AddOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(AddOperate,self).__init__(left,right,"add",args,name)

  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))
    part1= AddOperate.wrapper(self.left.partGrad(partial,prevOp),self.right.partGrad(partial,prevOp))
    rst=part1
    self._grads[partial.name]=rst
    return rst

  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "("+self.left.expression() + "+" + self.right.expression()+")"
    self._expressionStr = rst
    return rst
  
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst = dim.add(self.left.eval(),self.right.eval())
    self._data = rst
    return rst

  @staticmethod
  def wrapper(left,right,args=None,name=None):
    if (left.type=="Constant" and (left.data==0).all()): return right
    if (right.type=="Constant" and (right.data==0).all()): return left
    if (left.type=="Constant" and right.type=="Constant"): return Constant(dim.add(left.data,right.data))
    return AddOperate(left,right,args,name)

class SubOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(SubOperate,self).__init__(left,right,"sub",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))
    part1= SubOperate.wrapper(self.left.partGrad(partial,prevOp),self.right.partGrad(partial,prevOp))
    rst=part1
    self._grads[partial.name]=rst
    return rst

  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    part1= self.left.expression()
    part2= self.right.expression()
    if (part1=='0'):
      if (part2.slice(0,1)=='-'): rst= "{}".format(part2.slice(1))  
      else: rst= "(-{})".format(part2)
    elif (part2=='0'):
      rst= "{}".format(part1)
    else:
      rst = "({}-{})".format(part1,part2)

    self._expressionStr = rst
    return rst

  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst = dim.sub(self.left.eval(),self.right.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    if (right.type=="Constant" and (right.data==0).all()): return left
    if (left.type=="Constant" and right.type=="Constant"): return  Constant(dim.sub(left.data,right.data))
    if (left.isSame(right)): return Constant(1)
    
    return SubOperate(left,right,args,name)
    
class MulOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(MulOperate,self).__init__(left,right,"mul",args,name)
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))
    
    part1 = MulOperate.wrapper(self.right,prevOp)
    part2 = self.left.partGrad(partial,part1)
    part3 = MulOperate.wrapper(self.left,prevOp)
    part4 = self.right.partGrad(partial,part3)
    part5 = AddOperate.wrapper(part2,part4)
    rst = part5
    self._grads[partial.name]=rst
    return rst
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    part1=self.left.expression()
    part2=self.right.expression()
    if (part1=='-1'): rst= "-{}".format(part2)
    elif (part2=='-1'): rst= "-{}".format(part1)
    else: rst = "({}*{})".format(part1,part2)
    self._expressionStr = rst
    return rst
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst = dim.mul(self.left.eval(),self.right.eval())
    self._data = rst
    return rst
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    if (left.type=="Constant" and (left.data==0).all()): return Constant(0)
    if (right.type=="Constant" and (right.data==0).all()): return Constant(0)
    if (left.type=="Constant" and right.type=="Constant"): return Constant(dim.mul(left.data,right.data))
    if (left.type== "Constant" and (left.data==1).all()): return right
    if (right.type=="Constant" and (right.data==1).all()): return left
    
    return MulOperate(left,right,args,name)

class DivOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(DivOperate,self).__init__(left,right,"div",args,name)
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))
    part1 = MulOperate.wrapper(self.right,prevOp)
    part2 = self.left.partGrad(partial,part1)
    part3 = MulOperate.wrapper(self.left,prevOp)
    part4 = self.right.partGrad(partial,part3)
    part5 = SubOperate.wrapper(part2,part4)
    part6 = PowOperate.wrapper(self.right,Constant(2))
    part7 = DivOperate.wrapper(part5,part6)
    rst = part7
    self._grads[partial.name]=rst
    return rst  

  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    part1=self.left.expression()
    part2=self.right.expression()
    if (part2=='-1'): rst= "-{}".format(part1)
    else: rst = "({}/{})".format(part1,part2)
    self._expressionStr = rst
    return rst
  
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst = dim.div(self.left.eval(),self.right.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    if (left.type=="Constant" and (left.data==0).all()): return Constant(0)
    if (right.type=="Constant" and (right.data==0).all()): raise Exception("错误：除零的表达式") 
    if (left.type=="Constant" and right.type=="Constant"): return  Constant(dim.div(left.data,right.data))
    if (right.type=="Constant" and (right.data==1).all()): return left
    if (left.isSame(right)): return Constant(1)
    
    return DivOperate(left,right,args,name)

class PowOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(PowOperate,self).__init__(left,right,"pow",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))
    c = Constant(self.right.eval() - 1)
    part2 = PowOperate.wrapper(self.left,c)
    part3 = MulOperate.wrapper(self.right,part2)
    part4 = MulOperate.wrapper(part3,prevOp)
    rst = self.left.partGrad(partial,part4)
    self._grads[partial.name]=rst
    return rst  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst=self.left.expression() + "^" + self.right.expression()
    self._expressionStr = rst
    return rst
  
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst= dim.pow(self.left.eval(),self.right.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    if (right.type=="Constant" and (right.data==0).all()): return Constant(1)
    if (right.type=="Constant" and (right.data==1).all()): return left
    if (left.type=="Constant" and (left.data==1).all()): return Constant(1)
    return PowOperate(left,right,args,name)

class ExpOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(ExpOperate,self).__init__(left,right,"exp",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))

    part1 = MulOperate.wrapper(self,prevOp)
    part2 = self.left.partGrad(partial,part1)
    rst = part2
    self._grads[partial.name]=rst
    return rst

  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "e^"+self.left.expression()
    self._expressionStr = rst
    return rst

  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst= dim.exp(self.left.eval())
    self._data = rst
    return rst
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return ExpOperate(left,right,args,name)

class LogOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(LogOperate,self).__init__(left,right,"log",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads[partial.name]): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))

    part1 = DivOperate.wrapper(Constant(1),self.left)
    part2 = MulOperate.wrapper(part1,prevOp)
    part3 = self.left.partGrad(partial,part2)
    rst = part3
    self._grads[partial.name]=rst
    return rst

  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "ln("+self.left.expression()+")"
    self._expressionStr = rst
    return rst
  def eval(self):
    if (self.catch and self._data !=None): return self._data
    rst= dim.log(self.left.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return LogOperate(left,right,args,name)

class Log2Operate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(Log2Operate,self).__init__(left,right,"log2",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))

    part1 = DivOperate.wrapper(Constant(1/Math.log(2)),self.left)
    part2 = MulOperate.wrapper(part1,prevOp)
    part3 = self.left.partGrad(partial,part2)
    rst = part3
    self._grads[partial.name]=rst
    return rst

  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "log2("+self.left.expression()+")"
    self._expressionStr = rst
    return rst
  
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst= dim.log2(self.left.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return Log2Operate(left,right,args,name)

class Log10Operate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(Log10Operate,self).__init__(left,right,"log10",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))

    part1 = DivOperate.wrapper(Constant(1/Math.log(10)),self.left)
    part2 = MulOperate.wrapper(part1,prevOp)
    part3 = self.left.partGrad(partial,part2)
    rst = part3
    self._grads[partial.name]=rst
    return rst
  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "log10("+self.left.expression()+")"
    self._expressionStr = rst
    return rst
  
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst= dim.log10(self.left.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return Log10Operate(left,right,args,name)

class SinOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(SinOperate,self).__init__(left,right,"sin",args,name)
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))

    part1 = CosOperate.wrapper(self.left,None)
    part2 = MulOperate.wrapper(part1,prevOp)
    rst = self.left.partGrad(partial,part2)
    self._grads[partial.name]=rst
    return rst
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "sin("+self.left.expression()+")"
    self._expressionStr = rst
    return rst
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst= dim.sin(self.left.eval())
    self._data = rst
    return rst
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return SinOperate(left,right,args,name)

class CosOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(CosOperate,self).__init__(left,right,"cos",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))

    part1 = SinOperate.wrapper(self.left,None)
    part2 = MulOperate.wrapper(Constant(-1),part1)
    part3 = MulOperate.wrapper(part2,prevOp)
    rst = self.left.partGrad(partial,part3)
    self._grads[partial.name]=rst
    return rst
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "cos("+self.left.expression()+")"
    self._expressionStr = rst
    return rst

  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst= dim.cos(self.left.eval())
    self._data = rst
    return rst
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return CosOperate(left,right,args,name)

class TanOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(TanOperate,self).__init__(left,right,"tan",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))

    part1 = CosOperate.wrapper(self.left,None)
    part2 = PowOperate.wrapper(part1,2)
    part3 = DivOperate.Wrapper(Constant(1),part2)
    part4 = MulOperate.wrapper(part3,prevOp)
    rst = self.left.partGrad(partial,part4)
    self._grads[partial.name]=rst
    return rst
  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "tan("+self.left.expression()+")"
    self._expressionStr = rst
    return rst
  
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst= dim.tan(self.left.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(self,left,right,args=None,name=None):
    return TanOperate(left,right,args,name)
#Arc
class AsinOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(AsinOperate,self).__init__(left,right,"asin",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))

    part1 = PowOperate.wrapper(self.left,2)
    part2 = SubOperate.wrapper(Constant(1),part1)
    part3 = SqrtOperate.wrapper(part1,None)
    part4 = DivOperate.wrapper(Constant(1),part3)
    part5 = MulOperate.wrapper(part4,prevOp)
    rst = self.left.partGrad(partial,part5)
    self._grads[partial.name]=rst
    return rst
  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "asin("+self.left.expression()+")"
    self._expressionStr = rst
    return rst
  
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst= dim.asin(self.left.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return AsinOperate(left,right,args,name)

class AcosOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(AcosOperate,self).__init__(left,right,"acos",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))
    part1 = PowOperate.wrapper(self.left,2)
    part2 = SubOperate.wrapper(Constant(1),part1)
    part3 = SqrtOperate.wrapper(part1,None)
    part4 = DivOperate.wrapper(Constant(-1),part3)
    part5 = MulOperate.wrapper(part4,prevOp)
    rst = self.left.partGrad(partial,part5)
    self._grads[partial.name]=rst
    return rst
  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "acos("+self.left.expression()+")"
    self._expressionStr = rst
    return rst
  
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst= dim.acos(self.left.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return AcosOperate(left,right,args,name)

class AtanOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(AtanOperate,self).__init__(left,right,"atan",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))

    part1 = PowOperate.wrapper(self.left,2)
    part2 = AddOperate.wrapper(Constant(1),part1)
    part3 = DivOperate.wrapper(Constant(1),part2)
    part4 = MulOperate.wrapper(part3,prevOp)
    rst = self.left.partGrad(partial,part4)
    self._grads[partial.name]=rst
    return rst
  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "atan("+self.left.expression()+")"
    self._expressionStr = rst
    return rst

  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst= dim.atan(self.left.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return AtanOperate(left,right,args,name)
#h
class SinhOperate(Operate):
  def __init__(self,left,right,args,name):
    super(SinhOperate,self).__init__(self,left,right,"sinh",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))

    part1 = CoshOperate.wrapper(self.left,None)
    part2 = MulOperate.wrapper(part1,prevOp)
    rst = self.left.partGrad(partial,part2)
    self._grads[partial.name]=rst
    return rst
  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "sinh("+self.left.expression()+")"
    self._expressionStr = rst
    return rst
  
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst= dim.sinh(self.left.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return SinhOperate(left,right,args,name)

class CoshOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(CoshOperate,self).__init__(left,right,"cosh",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))

    part1 = SinhOperate.wrapper(self.left,None)
    part2 = MulOperate.wrapper(part1,prevOp)
    rst = self.left.partGrad(partial,part2)
    self._grads[partial.name]=rst
    return rst
  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "cosh("+self.left.expression()+")"
    self._expressionStr = rst
    return rst
  
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst= dim.cosh(self.left.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return CoshOperate(left,right,args,name)

class TanhOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(TanhOperate,self).__init__(left,right,"tanh",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))

    part1 = CoshOperate.wrapper(self.left,None)
    part2 = PowOperate.wrapper(part1,2)
    part3 = DivOperate.wrapper(Constant(1),part2)
    part4 = MulOperate.wrapper(part3,prevOp)
    rst = self.left.partGrad(partial,part4)
    self._grads[partial.name]=rst
    return rst
  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "tanh("+self.left.expression()+")"
    self._expressionStr = rst
    return rst
  
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst= dim.tanh(self.left.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return TanhOperate(left,right,args,name)

#arch
class AsinhOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(left,right,"asinh",args,name)
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))

    raise Exception("not impleted!")
    self._grads[partial.name]=rst
    return rst
  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "asinh("+self.left.expression()+")"
    self._expressionStr = rst
    return rst
  
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst= dim.asinh(self.left.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return AsinhOperate(left,right,args,name)

class AcoshOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(AcoshOperate,self).__init__(left,right,"acosh",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))

    raise Exception("not impleted!")

    self._grads[partial.name]=rst
    return rst
  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "acosh("+self.left.expression()+")"
    self._expressionStr = rst
    return rst
  
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst= dim.acosh(self.left.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return AcoshOperate(left,right,args,name)

class AtanhOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(AtanhOperate,self).__init__(left,right,"atanh",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))


    raise Exception("not impleted!")

    self._grads[partial.name]=rst
    return rst
  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "atanh("+self.left.expression()+")"
    self._expressionStr = rst
    return rst
  
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst= dim.atanh(self.left.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return AtanhOperate(left,right,args,name)

class SumOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(SumOperate,self).__init__(left,right,"sum",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))

    part1 = Constant(dim.ones(self.left.eval().shape))
    part2 = MulOperate.wrapper(part1,prevOp)
    part3 = self.left.partGrad(partial,part2)
    rst = part3
    self._grads[partial.name]=rst
    return rst

  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "sum("+self.left.expression()+")"
    self._expressionStr = rst
    return rst
    
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst= dim.sum(self.left.eval())
    self._data = rst
    return rst

  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return SumOperate(left,right,args,name)

class MeanOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(MeanOperate,self).__init__(left,right,"mean",args,name)
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))

    part1 = Constant(dim.fill(1/self.left.eval().size,self.left.eval().shape))
    part2 = MulOperate.wrapper(part1,prevOp)
    part3 = self.left.partGrad(partial,part2)
    rst = part3

    self._grads[partial.name]=rst
    return rst
  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "mean("+self.left.expression()+")"
    self._expressionStr = rst
    return rst
  
  def eval(self):
    if (self.catch and self._data !=None): return self._data
    rst= dim.mean(self.left.eval())
    self._data = rst
    return rst

  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return MeanOperate(left,right,args,name)

class MaxOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(MaxOperate,self).__init__(left,right,"max",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))
    zeros = dim.zeros(self.left.eval().shape)
    zeros.reshape(zeros.size)[self.args["indices"]]=1
    part1 =Constant(zeros)
    part2 = MulOperate.wrapper(part1,prevOp)
    part3 = self.left.partGrad(partial,part2)
    rst = part3
    self._grads[partial.name]=rst
    return rst
  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "max("+self.left.expression()+")"
    self._expressionStr = rst
    return rst
  
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst= dim.max(self.left.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return MaxOperate(left,right,args,name)

class MinOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(MinOperate,self).__init__(left,right,"min",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))
    zeros = dim.zeros(self.left.eval().shape)
    zeros.reshape(zeros.size)[self.args["indices"]]=1
    part1 =Constant(zeros)
    part2 = MulOperate.wrapper(part1,prevOp)
    part3 = self.left.partGrad(partial,part2)
    rst = part3
    self._grads[partial.name]=rst
    return rst
  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "min("+self.left.expression()+")"
    self._expressionStr = rst
    return rst
  
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst= dim.min(self.left.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return MinOperate(left,right,args,name)

class AbsOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(AbsOperate,self).__init__(left,right,"abs",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))

    pre1 = AbsOperate.wrapper(self.left,None)
    part1 = DivOperate.wrapper(self.left,pre1)
    part2 = MulOperate.wrapper(part1,prevOp)
    part3 = self.left.partGrad(partial,part2)
    rst = part3
    self._grads[partial.name]=rst
    return rst
  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "|"+self.left.expression()+"|"
    self._expressionStr = rst
    return rst
  
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst= dim.abs(self.left.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return AbsOperate(left,right,args,name)

class DotOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(DotOperate,self).__init__(left,right,"dot",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))
    '''直接求eval的方式也没有问题，因为上一层的反算结果已经完成了。这种方式的问题是
       不能在总算式的gradFn.gradExpression()中看到各偏导的dot计算公式
    '''
    dLeft = Constant(prevOp.eval().dot(self.right.eval().T))
    dRight = Constant(self.left.eval().T.dot(prevOp.eval()))
    tRight = TOperate.wrapper(self.right,None)
    dLeft = DotOperate.wrapper(prevOp,tRight)
    tLeft = TOperate.wrapper(self.left,None)
    dRight = DotOperate.wrapper(tLeft,prevOp)
    if (self.left.name==partial.name):
      part1 = dLeft
      part2 = self.right.partGrad(partial,dRight)
    elif (self.right.name==partial.name):
      part1 = self.left.partGrad(partial,dLeft)
      part2 = dRight
    else:
      part1=self.left.partGrad(partial,dLeft)
      part2=self.right.partGrad(partial,dRight)

    part3=AddOperate.wrapper(part1,part2)
    rst = part3
    self._grads[partial.name]=rst
    return rst  

  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    part1=self.left.expression()
    part2=self.right.expression()
    rst = "({}@{})".format(part1,part2)
    self._expressionStr = rst
    return rst
 
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst = dim.dot(self.left.eval(),self.right.eval())
    self._data = rst
    return rst

  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return DotOperate(left,right,args,name)

class TOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(TOperate,self).__init__(left,right,"T",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))
    
    if (self.left.name==partial.name):
      part1 = TOperate.wrapper(prevOp,None)
      rst = part1
    else:
      part1 = TOperate.wrapper(prevOp,None)
      part2 = self.left.partGrad(partial,part1)
      part3 = TOperate.wrapper(part2,None)
      rst = part3
    
    self._grads[partial.name]=rst
    return rst  
  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "T("+self.left.expression()+")"
    self._expressionStr = rst
    return rst
   
  def eval(self):
    if (self.catch and self._data is not None): return self._data

    rst=self.left.eval()
    if (not self.isNumber(rst)): rst=rst.t()
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return TOperate(left,right,args,name)


class ReluOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(ReluOperate,self).__init__(left,right,"relu",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))
    
    part1 = ReluDeriOperate.wrapper(self.left,None)
    part2 = MulOperate.wrapper(part1,prevOp)
    part3 = self.left.partGrad(partial,part2)
    rst = part3
    self._grads[partial.name]=rst
    return rst  

  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "relu("+self.left.expression()+")"
    self._expressionStr = rst
    return rst
 
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst = dim.nn.functional.relu(self.left.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return ReluOperate(left,right,args,name)

class ReluDeriOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(ReluDeriOperate,self).__init__(left,right,"reluDeri",args,name)

  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))
    
    raise Exception('not implemented')

    self._grads[partial.name]=rst
    return rst  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "reluDeri("+self.left.expression()+")"
    self._expressionStr = rst
    return rst 
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst = dim.nn.functional.reluDeri(self.left.eval())
    self._data = rst
    return rst

  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return ReluDeriOperate(left,right,args,name)

class SigmoidOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(SigmoidOperate,self).__init__(left,right,"sigmoid",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))
    
    part1 = SigmoidDeriOperate.wrapper(self.left,None)
    part2 = MulOperate.wrapper(part1,prevOp)
    part3 = self.left.partGrad(partial,part2)
    rst = part3
    self._grads[partial.name]=rst
    return rst  
  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "sigmoid("+self.left.expression()+")"
    self._expressionStr = rst
    return rst 
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst = dim.nn.functional.sigmoid(self.left.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return SigmoidOperate(left,right,args,name)

class SigmoidDeriOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(SigmoidDeriOperate,self).__init__(left,right,"sigmoidDeri",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))
    
    raise Exception('not implemented')

    self._grads[partial.name]=rst
    return rst  
  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "sigmoidDeri("+self.left.expression()+")"
    self._expressionStr = rst
    return rst 
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst = dim.nn.functional.sigmoidDeri(self.left.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return SigmoidDeriOperate(left,right,args,name)
    
class SoftmaxOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(SoftmaxOperate,self).__init__(left,right,"softmax",args,name)
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))
    
    part1 = SoftmaxDeriOperate.wrapper(self,prevOp)
    part2 = self.left.partGrad(partial,part1)
    rst = part2        
    self._grads[partial.name]=rst
    return rst  
  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "softmax("+self.left.expression()+")"
    self._expressionStr = rst
    return rst

  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst = dim.nn.functional.softmax(self.left.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return SoftmaxOperate(left,right,args,name)

class SoftmaxDeriOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(SoftmaxDeriOperate,self).__init__(left,right,"softmaxDeri",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))
    
    raise Exception('not implemented')

    self._grads[partial.name]=rst
    return rst  

  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "softmaxDeri("+self.left.expression()+")"
    self._expressionStr = rst
    return rst
  
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst = dim.nn.functional.softmaxDeri(self.left.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return SoftmaxDeriOperate(left,right,args,name)

class CrossEntropyOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(CrossEntropyOperate,self).__init__(left,right,"crossEnropty",args,name)

  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))
    
    part1 = CrossEntropyDeriOperate.wrapper(self.left,self.right)
    part2 = MulOperate.wrapper(part1,prevOp)
    part3 = self.left.partGrad(partial,part2)
    rst = part3        
    self._grads[partial.name]=rst
    return rst  
  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "crossEntropy("+self.left.expression()+","+self.right.expression()+")"
    self._expressionStr = rst
    return rst
   
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst = dim.nn.functional.crossEntropy(self.left.eval(),self.right.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return CrossEntropyOperate(left,right,args,name)

class CrossEntropyDeriOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(CrossEntropyDeriOperate,self).__init__(left,right,"crossEntropyDeri",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))
    
    raise Exception('not implemented')
    
    self._grads[partial.name]=rst
    return rst  
  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "crossEntropyDeri("+self.left.expression()+","+self.right.expression()+")"
    self._expressionStr = rst
    return rst 
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst = dim.nn.functional.crossEntropyDeri(self.left.eval(),self.right.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return CrossEntropyDeriOperate(left,right,args,name)

class MSELossOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(MSELossOperate,self).__init__(left,right,"mseLoss",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))
    
    part1 = MulOperate.wrapper(self.left,prevOp)
    part2 = MulOperate.wrapper(self.right,prevOp)
    part3 = self.left.partGrad(partial,part1)
    part4 = self.right.partGrad(partial,part2)
    rst = AddOperate.wrapper(part3,part4)        
    self._grads[partial.name]=rst
    return rst  
  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "mseLoss("+self.left.expression()+")"
    self._expressionStr = rst
    return rst
   
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst = dim.nn.functional.mseLosss(self.left.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return MSELossOperate(left,right,args,name)

class Conv1dOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(Conv1dOperate,self).__init__(left,right,"conv1d",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))
    
    dLeft=ConvTranspose1dOperate.wrapper(prevOp,self.right,self.args)
    temp1 = prevOp.eval().swapaxes(0,1)
    temp2 = self.left.eval().swapaxes(0,1)
    temp3 = dim.nn.functional.conv1d(temp2,temp1)
    dRight = Constant(temp3.swapaxes(0,1))
    if (self.left.name==partial.name):
      part1 = dLeft
      part2=self.right.partGrad(partial,dRight)
    elif (self.right.name==partial.name):
      part1 = dRight
      part2=self.left.partGrad(partial,dLeft)
    else:
      part1=self.left.partGrad(partial,dLeft)
      part2=self.right.partGrad(partial,dRight)
  
    part3=AddOperate.wrapper(part1,part2)
    rst = part3
    self._grads[partial.name]=rst
    return rst  
  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "conv1d("+self.left.expression()+","+self.right.expression()+")"
    self._expressionStr = rst
    return rst
  
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst=dim.nn.functional.conv1d(self.left.eval(),self.right.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return Conv1dOperate(left,right,args,name)

class Conv2dOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(Conv2dOperate,self).__init__(left,right,"conv2d",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))
    
    dLeft=ConvTranspose2dOperate.wrapper(prevOp,self.right,self.args)
    temp1 = prevOp.eval().swapaxes(0,1)
    temp2 = self.left.eval().swapaxes(0,1)
    temp3 = dim.nn.functional.conv2d(temp2,temp1)
    dRight = Constant(temp3.swapaxes(0,1))
    if (self.left.name==partial.name):
      part1 = dLeft
      part2=self.right.partGrad(partial,dRight)
    elif (self.right.name==partial.name):
      part1 = dRight
      part2=self.left.partGrad(partial,dLeft)
    else:
      part1=self.left.partGrad(partial,dLeft)
      part2=self.right.partGrad(partial,dRight)

    part3=AddOperate.wrapper(part1,part2)
    rst = part3
    self._grads[partial.name]=rst
    return rst  
  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "conv2d("+self.left.expression()+","+self.right.expression()+")"
    self._expressionStr = rst
    return rst

  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst=dim.nn.functional.conv2d(self.left.eval(),self.right.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return Conv2dOperate(left,right,args,name)

class ConvTranspose1dOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(ConvTranspose1dOperate,self).__init__(left,right,"convTranspose1d",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))
    
    raise Exception("not implemented")
    
    self._grads[partial.name]=rst
    return rst  
  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "convTranspose1d("+self.left.expression()+","+self.right.expression()+")"
    self._expressionStr = rst
    return rst
   
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst=dim.nn.functional.convTranspose1d(self.left.eval(),self.right.eval(),1,self.args["padding"])
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return ConvTranspose1dOperate(left,right,args,name)

class ConvTranspose2dOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(ConvTranspose2dOperate,self).__init__(left,right,"convTranspose2d",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))
    
    raise Exception("not implemented")
    self._grads[partial.name]=rst
    return rst  
  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "convTranspose2d("+self.left.expression()+","+self.right.expression()+")"
    self._expressionStr = rst
    return rst
   
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst=dim.nn.functional.convTranspose2d(self.left.eval(),self.right.eval(),1,self.args["padding"])
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return ConvTranspose2dOperate(left,right,args,name)

class MaxPool1dOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(MaxPool1dOperate,self).__init__(left,right,"maxPool1d",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))
    
    part1 = MaxUnpool1dOperate.wrapper(prevOp,Constant(self.right.eval()),self.args)
    part2 = self.left.partGrad(partial,part1)
    rst = part2       
    self._grads[partial.name]=rst
    return rst  
  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "maxPool1d("+self.left.expression()+","+self.right.expression()+")"
    self._expressionStr = rst
    return rst 
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst=dim.nn.functional.maxPool1d(self.left.eval(),self.right.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return MaxPool1dOperate(left,right,args,name)

class AvgPool1dOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(AvgPool1dOperate,self).__init__(left,right,"avgPool1d",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))
    
    part1 = AvgUnpool1dOperate.wrapper(prevOp,Constant(self.right.eval()))
    part2 = self.left.partGrad(partial,part1)
    rst = part2       
    self._grads[partial.name]=rst
    return rst  
  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "avgPool1d("+self.left.expression()+","+self.right.expression()+")"
    self._expressionStr = rst
    return rst 
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst=dim.nn.functional.avgPool1d(self.left.eval(),self.right.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return AvgPool1dOperate(left,right,args,name)

class MaxPool2dOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(MaxPool2dOperate,self).__init__(left,right,"maxPool2d",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))
    
    part1 = MaxUnpool2dOperate.wrapper(prevOp,Constant(self.right.eval()),self.args)
    part2 = self.left.partGrad(partial,part1)
    rst = part2       
    self._grads[partial.name]=rst
    return rst  
  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "maxPool2d("+self.left.expression()+","+self.right.expression()+")"
    self._expressionStr = rst
    return rst
  
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst=dim.nn.functional.maxPool2d(self.left.eval(),self.right.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return MaxPool2dOperate(left,right,args,name)

class AvgPool2dOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(AvgPool2dOperate,self).__init__(left,right,"avgPool2d",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))
    
    part1 = AvgUnpool2dOperate.wrapper(prevOp,Constant(self.right.eval()))
    part2 = self.left.partGrad(partial,part1)
    rst = part2       
    self._grads[partial.name]=rst
    return rst  
  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "avgPool2d("+self.left.expression()+","+self.right.expression()+")"
    self._expressionStr = rst
    return rst 
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst=dim.nn.functional.avgPool2d(self.left.eval(),self.right.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return AvgPool2dOperate(left,right,args,name)

class MaxUnpool1dOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(MaxUnpool1dOperate,self).__init__(left,right,"maxUnpool1d",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))
    
    raise Exception("not implemented")
    
    self._grads[partial.name]=rst
    return rst  
  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "maxUnpool1d("+self.left.expression()+","+self.right.expression()+")"
    self._expressionStr = rst
    return rst
 
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst=dim.nn.functional.maxUnpool1d(
        self.left.eval(),self.args["indices"],self.right.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return MaxUnpool1dOperate(left,right,args,name)

class AvgUnpool1dOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(AvgUnpool1dOperate,self).__init__(left,right,"avgUnpool1d",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))
    
    raise Exception("not implemented")

    self._grads[partial.name]=rst
    return rst  
  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "avgUnpool1d("+self.left.expression()+","+self.right.expression()+")"
    self._expressionStr = rst
    return rst
 
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst=dim.nn.functional.avgUnpool1d(self.left.eval(),self.right.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return AvgUnpool1dOperate(left,right,args,name)

class MaxUnpool2dOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(MaxUnpool2dOperate,self).__init__(left,right,"maxUnpool2d",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))
    
    raise Exception("not implemented")

    self._grads[partial.name]=rst
    return rst  
  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "maxUnpool2d("+self.left.expression()+","+self.right.expression()+")"
    self._expressionStr = rst
    return rst
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst=dim.nn.functional.maxUnpool2d(self.left.eval(),self.args["indices"],self.right.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return MaxUnpool2dOperate(left,right,args,name)

class AvgUnpool2dOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(AvgUnpool2dOperate,self).__init__(left,right,"avgUnpool2d",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))
    
    raise Exception("not implemented")

    self._grads[partial.name]=rst
    return rst  
  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "avgUnpool2d("+self.left.expression()+","+self.right.expression()+")"
    self._expressionStr = rst
    return rst
   
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst=dim.nn.functional.avgUnpol2d(self.left.eval(),self.right.eval())
    self._data = rst
    return rst

  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return AvgUnpool2dOperate(left,right,args,name)
