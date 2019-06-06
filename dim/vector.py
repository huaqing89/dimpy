#coding:utf-8
import numpy as np
import dim

class Vector(np.ndarray):
  def __init__(self,shape,buffer,dtype):
    super(Vector,self).__init__()
    self.__gradFn=None
    self.__grad=None
    self.__requiresGrad=False
  def ensureVector(self,a,dtype='float32'):
    if isinstance(a,Vector): return a
    if isinstance(a,(int,float)): 
      a=np.array([a],dtype)
    elif isinstance(a,list):
      a=np.array(a,dtype)
    return Vector(shape=a.shape,buffer=a,dtype=a.dtype.name)
  
  @property
  def requiresGrad(self):
    try:
      return self.__requiresGrad
    except:
      return False
  @requiresGrad.setter
  def requiresGrad(self,b):
    self.__requiresGrad=b
  
  @property
  def gradFn(self):
    try:
      return self.__gradFn
    except:
      return None
  @gradFn.setter
  def gradFn(self,op):
    self.__gradFn=op

  @property
  def grad(self):
    try:
      return self.__grad
    except:
      return None
  @grad.setter
  def grad(self,a):
    self.__grad=a

  def setGradFn(self,rst,opStr,**kwargs):
    left = kwargs.get("left",self)
    right = kwargs.get("right",None)
    args = kwargs.get("args",None)
    name = kwargs.get("name",None)
    if (isinstance(left,Vector) and left.requiresGrad) or (isinstance(right,Vector) and right.requiresGrad):
      #print("setGradFn",left.shape,opStr)
      rst.requiresGrad=True
      if left is None : leftFn=None
      elif getattr(left,"gradFn",None): leftFn=left.gradFn
      else: leftFn=dim.autograd.Constant(left)
      
      if right is None : rightFn=None
      elif getattr(right,"gradFn",None): rightFn=right.gradFn
      else: rightFn=dim.autograd.Constant(right)
      #print("left",left,leftFn)
      #print("right",right,rightFn)
      rst.gradFn=dim.autograd.Operate.wrapper(leftFn,rightFn,opStr,args,name)
      rst.gradFn._data=rst.view() #减少一次求值操作 gradFn.eval()将使用gradFn._data作为catch
    return rst   

  def __add__(self,other): 
    rst = super(Vector,self).__add__(other)
    rst = self.setGradFn(rst,"add",right=other)
    return rst
  def __radd__(self,other):
    return self+other
  def __sub__(self,other): 
    rst = super(Vector,self).__sub__(other)
    rst = self.setGradFn(rst,"sub",right=other)
    return rst
  def __rsub__(self,other):
    return -self+other
  def __mul__(self,other): 
    rst = super(Vector,self).__mul__(other)
    rst = self.setGradFn(rst,"mul",right=other)
    return rst
  def __rmul__(self,other):
    return self*other
  def __truediv__(self,other): 
    rst = super(Vector,self).__truediv__(other)
    rst = self.setGradFn(rst,"div",right=other)
    return rst
  def __rtruediv__(self,other):
    return self**(-1)*other
  def __pow__(self,n): 
    rst = super(Vector,self).__pow__(n)
    rst = self.setGradFn(rst,"pow",right=n)
    return rst  
  def __neg__(self): 
    rst = super(Vector,self).__neg__()
    rst = self.setGradFn(rst,"sub",left=0,right=self)
    return rst  
  
  #def __getitem__(self,index): return super(Vector,self).__getitem__(index)
  
  #def __gt__(self,other): return super(Vector,self).__gt__(other)
  #def __lt__(self,other): return super(Vector,self).__lt__(other)
  #def __ge__(self,other): return super(Vector,self).__ge__(other)
  #def __le__(self,other): return super(Vector,self).__le__(other)
  #def __eq__(self,other): return super(Vector,self).__eq__(other)
  #def __ne__(self,other): return super(Vector,self).__ne__(other)

  def float(self):
    rst=self.ensureVector(self.astype('float32'))
    return rst
  def double(self):
    rst=self.ensureVector(self.astype('float64'))
    return rst
  def int(self):
    rst=self.ensureVector(self.astype('int32'))
    return rst
  def long(self):
    rst=self.ensureVector(self.astype('int64'))
    return rst
  def radians(self):
    rst=self.ensureVector(np.radians(self))
    return rst
  def sin(self):
    rst=self.ensureVector(np.sin(self))
    rst=self.setGradFn(rst,"sin")
    return rst
  def cos(self):
    rst=self.ensureVector(np.cos(self))
    rst=self.setGradFn(rst,"cos")
    return rst
  def tan(self):
    rst=self.ensureVector(np.tan(self))
    rst=self.setGradFn(rst,"tan")
    return rst
  def asin(self):
    rst=self.ensureVector(np.asin(self))
    rst=self.setGradFn(rst,"asin")
    return rst
  def acos(self):
    rst=self.ensureVector(np.acos(self))
    rst=self.setGradFn(rst,"acos")
    return rst
  def atan(self):
    rst=self.ensureVector(np.atan(self))
    rst=self.setGradFn(rst,"atan")
    return rst
  def sinh(self):
    rst=self.ensureVector(np.sinh(self))
    rst=self.setGradFn(rst,"sinh")
    return rst
  def cosh(self):
    rst=self.ensureVector(np.cosh(self))
    rst=self.setGradFn(rst,"cosh")
    return rst
  def tanh(self):
    rst=self.ensureVector(np.tanh(self))
    rst=self.setGradFn(rst,"tanh")
    return rst
  def asinh(self):
    rst=self.ensureVector(np.asinh(self))
    rst=self.setGradFn(rst,"asinh")
    return rst
  def acosh(self):
    rst=self.ensureVector(np.acosh(self))
    rst=self.setGradFn(rst,"acosh")
    return rst
  def atanh(self):
    rst=self.ensureVector(np.atanh(self))
    rst=self.setGradFn(rst,"atanh")
    return rst

  def log(self):
    rst=self.ensureVector(np.log(self))
    rst=self.setGradFn(rst,"log")
    return rst
  def log2(self): 
    rst=self.ensureVector(np.log2(self))
    rst=self.setGradFn(rst,"log2")
    return rst
  def log10(self): 
    rst=self.ensureVector(np.log10(self))
    rst=self.setGradFn(rst,"log10")
    return rst
  def exp(self):
    rst=self.ensureVector(np.exp(self))
    rst=self.setGradFn(rst,"exp")
    return rst
  def sqrt(self): 
    rst=self**0.5
    rst=self.setGradFn(rst,"pow",right=0.5)
    return rst
  def square(self): 
    rst=self ** 2
    rst=self.setGradFn(rst,"pow",right=2)
    return rst
  def pow(self,n):
    rst= self**n
    rst= self.setGradFn(rst,"pow",right=n)
    return rst
  def floor(self): 
    rst=self.ensureVector(np.floor(self))
    rst=self.setGradFn(rst,"floor")
    return rst
  def ceil(self): 
    rst=self.ensureVector(np.ceil(self))
    return rst
  def around(self,n): 
    rst=self.ensureVector(np.around(self,n))
    return rst
  def abs(self):
    rst=self.ensureVector(np.abs(self))
    rst=self.setGradFn(rst,"abs")
    return rst
  def neg(self): 
    return -self
  def reciprocal(self): 
    rst=self.ensureVector(np.reciprocal(self))
    rst=self.setGradFn(rst,"div",left=1,right=self)
    return rst


  def add(self,a): return self+a
  def sub(self,a): return self-a
  def mul(self,a): return self*a
  def div(self,a): return self/a

  def mod(self,b): return self.ensureVector(np.neg(self,b))
  def subtract(self,b): return self.sub(b)
  def multiply(self,b): return self.mul(b)
  def divide(self,b):   return self.div(b)
  def negative(slef): return self.neg()
  def power(self,n):  return self.pow(n)
  
  def sign(self): return self.ensureVector(np.sign(self))
  def gt(self,a): return self>a
  def lt(self,a): return self<a
  def ge(self,a): return self>=a
  def le(self,a): return self<=a
  def eq(self,a): return self==a
  def ne(self,a): return self!=a
  def allclose(self,a): return self.ensureVector(np.allclose(self,a))
  def all(self,axis=None): 
    return self.ensureVector(super(Vector,self).all(axis))
  def any(self,axis=None): 
    return self.ensureVector(super(Vector,self).any(axis))

  def sum(self,axis=None,**kwargs):
    rst = super(Vector,self).sum(axis,**kwargs)
    if (axis==None):
      rst=self.setGradFn(rst,"sum")
    return rst
  def mean(self,axis=None,**kwargs):
    rst = super(Vector,self).mean(axis,**kwargs)
    if (axis==None):
      rst=self.setGradFn(rst,"mean")
    return rst
  def max(self,axis=None,**kwargs):
    rst = super(Vector,self).max(axis,**kwargs)
    if (axis==None):
      rst=self.setGradFn(rst,"max",args={"indices":self.argmax()})
    return rst
  def min(self,axis=None,**kwargs):
    rst = super(Vector,self).min(axis,**kwargs)
    if (axis==None):
      rst=self.setGradFn(rst,"min",args={"indices":self.argmin()})
    return rst
  def argmax(self,axis=None,**kwargs):
    return super(Vector,self).argmax(axis,**kwargs)
  def argmin(self,axis=None,**kwargs):
    return super(Vector,self).argmin(axis,**kwargs)
  def var(self,axis=None,**kwargs):
    rst = super(Vector,self).var(axis,**kwargs)
    if (axis==None):
      rst=self.setGradFn(rst,"var")
    return rst
  def std(self,axis=None,**kwargs):
    rst = super(Vector,self).std(axis,**kwargs)
    if (axis==None):
      rst=self.setGradFn(rst,"std")
    return rst
  def cov(self,axis=None,**kwargs):
    rst = super(Vector,self).cov(axis,**kwargs)
    if (axis==None):
      rst=self.setGradFn(rst,"cov")
    return rst
  def ptp(self,axis=None,**kwargs):
    rst = super(Vector,self).ptp(axis,**kwargs)
    if (axis==None):
      rst=self.setGradFn(rst,"ptp")
    return rst
  def median(self,axis=None,**kwargs):
    rst = super(Vector,self).median(axis,**kwargs)
    if (axis==None):
      rst=self.setGradFn(rst,"median")
    return rst

  def dot(self,a):
    rst = super(Vector,self).dot(a)
    rst=self.setGradFn(rst,"dot",right=a)
    return rst
    
  def t(self):
    rst = self.T.copy()
    #rst=self.setGradFn(rst,"T")
    return rst
    
  def rot180(self):
    a = self.reshape(self.size)
    a = a[::-1]
    a = a.reshape(self.shape)
    return a
  def onehotEncode(self,n=None):
    a=self
    if (a.ndim==1): a=a.reshape(a.size,1)
    if (a.ndim!=2 or a.shape[1]!=1): 
      raise Exception("对象要求是一维向量，或是n*1矩阵")
    max=int(a.max().value()+1)
    if (n is None): n=max
    if (n<max): n=max
    b=np.zeros((a.shape[0],n))
    for i,x in enumerate(a[:,0].tolist()):
      b[i,int(x)]=1
    return self.ensureVector(b)
     
  def labelEncode(self,dic=None):
    a=self
    if (a.ndim==1): a=a.reshape(a.size,1)
    if (a.ndim!=2 or a.shape[1]!=1): 
      raise Exception("对象要求是一维向量，或是n*1矩阵")
    b=a.copy()
    if (dic is None):
      v=list(set(b.flat))
      v.sort()
      dic={}
      for i,x in enumerate(v):dic[x]=i
    for x in dic:
      b[b==x]=dic[x]  
    return b.int()

  def pad(self,pad_width,mode="constant"):
    return np.pad(self,pad_width,mode=mode)
  
  def kron(self,b):
    return np.kron(self,b)
  def zNormal(self,axis=0):
    mean = self.mean(axis)
    std = self.std(axis)
    return (self-mean) / std
  def minmaxScalar(self,axis=0):
    dmin = self.min(axis)
    sub =self.max(axis) - dmin
    return (self - dmin)/ sub
  def maxabsScalar(self,axis=0):
    return self/self.max(axis).abs()  
  
  def clip(self,m,n): return np.clip(self,m,n)
  
  def hsplit(self,m): return np.split(self,m,1)
  def vsplit(self,m): return np.split(self,m,0)
  def split(self,m,axis=0): return np.split(self,m,axis)

  def take(self,axis,p): return np.take(self,axis,p)
        
  def value(self): return self.tolist()
  def numpy(self): return self.base
  
  def setGrad(self,bool=True):
    self.requiresGrad=bool
    self.grad=None
    self.gradFn=None
    if (bool and self.isLeaf):
      self.gradFn=dim.autograd.Variable(self)
  
  @property
  def isLeaf(self):
    return (not isinstance(self.gradFn,dim.autograd.Operate))

  def expression(self): 
    if (not self.gradFn): return None
    return self.gradFn.expression()
  def gradExpression(self):
    if (not self.gradFn): return None
    return self.gradFn.gradExpression()
  def backward(self,prevOp=None):
    if (not self.requiresGrad): raise Exception("after call setGrad(true) ,then use this function")
    if (prevOp): prevOp = dim.autograd.Constant(prevOp)
    variables=self.gradFn.variables()
    for v in variables:
      op=self.gradFn.backward(prevOp,v)
      grad=op.eval()
      v.data.grad = v.data.grad.add(grad) if v.data.grad is not None else grad
  def gradClear(self):
    self.gradFn.clearData()
    self.grad=None

  #define opencl
  def to_device(self):
    return dim.cl.to_device(self)
  def cl(self):
    return self.to_device()
  def cl_(self):
    print("??")
    self=dim.cl.to_device(self)
    return self