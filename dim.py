
'''
  kron(a,indices=false,dir="v"){
    return this.data.map((x,i)=>{
      if (x instanceof Vector) return x.kron(a,indices,"h")
      if (indices){ 
        return a.data[i].mul(x)
      }else{
        return a.mul(x)
      }
    }).reduce((m,n)=>{
      if (dir=="h") {
        if (m.ndim==1) return new Vector(m.data.concat(n.data))
        return m.hstack(n)
      }
      if (dir=="v") {
        if (m.ndim==1) return new Vector(m.data.concat(n.data))
        return m.vstack(n)
      }
    })
  }
}
'''
#coding:utf-8
import numpy as np
from nn import NN,Optimizer
import autograd 

class Vector(np.ndarray):
  def __init__(self,shape,buffer,dtype):
    super(Vector,self).__init__()
    self.__gradFn=None
    self.__grad=None
    self.__requiresGrad=False
  def ensureVector(self,a,dtype='float32'):
    if isinstance(a,Vector): return a
    if isinstance(a,int) or isinstance(a,float): 
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
      rst.requiresGrad=True
      if left is None : leftFn=None
      elif getattr(left,"gradFn",None): leftFn=left.gradFn
      else: leftFn=autograd.Constant(left)
      
      if right is None : rightFn=None
      elif getattr(right,"gradFn",None): rightFn=right.gradFn
      else: rightFn=autograd.Constant(right)
      #print("left",left,leftFn)
      #print("right",right,rightFn)
      rst.gradFn=autograd.Operate.wrapper(leftFn,rightFn,opStr,args,name)
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
    rst=self.ensureVector(np.sqrt(self))
    rst=self.setGradFn(rst,"pow",right=0.5)
    return rst
  def square(self): 
    rst=self.ensureVector(np.square(self))
    rst=self.setGradFn(rst,"pow",right=2)
    return rst
  def pow(self,n):
    return self**n
  def floor(self): 
    rst=self.ensureVector(np.floor(self))
    rst=self.setGradFn(rst,"floor")
    return rst
  def ceil(self): 
    rst=self.ensureVector(np.ceil(self))
    rst=self.setGradFn(rst,"ceil")
    return rst
  def around(self,n): 
    rst=self.ensureVector(np.around(self,n))
    rst=self.setGradFn(rst,"around",n)
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


  def add(self,a): print("add");return self+a
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

  def sum(self,axis=None):
    rst = super(Vector,self).sum(axis)
    if (axis==None):
      rst=self.setGradFn(rst,"sum")
    return rst
  def mean(self,axis=None):
    rst = super(Vector,self).mean(axis)
    if (axis==None):
      rst=self.setGradFn(rst,"mean")
    return rst
  def max(self,axis=None):
    rst = super(Vector,self).max(axis)
    if (axis==None):
      rst=self.setGradFn(rst,"max",args={"indices":self.argmax()})
    return rst
  def min(self,axis=None):
    rst = super(Vector,self).min(axis)
    if (axis==None):
      rst=self.setGradFn(rst,"min",args={"indices":self.argmin()})
    return rst
  def argmax(self,axis=None):
    return super(Vector,self).argmax(axis)
  def argmin(self,axis=None):
    return super(Vector,self).argmin(axis)
  def var(self,axis=None):
    rst = super(Vector,self).var(axis)
    if (axis==None):
      rst=self.setGradFn(rst,"var")
    return rst
  def std(self,axis=None):
    rst = super(Vector,self).std(axis)
    if (axis==None):
      rst=self.setGradFn(rst,"std")
    return rst
  def cov(self,axis=None):
    rst = super(Vector,self).cov(axis)
    if (axis==None):
      rst=self.setGradFn(rst,"cov")
    return rst
  def ptp(self,axis=None):
    rst = super(Vector,self).ptp(axis)
    if (axis==None):
      rst=self.setGradFn(rst,"ptp")
    return rst
  def median(self,axis=None):
    rst = super(Vector,self).median(axis)
    if (axis==None):
      rst=self.setGradFn(rst,"median")
    return rst

  def dot(self,a):
    rst = super(Vector,self).dot(a)
    rst=self.setGradFn(rst,"dot",right=a)
    return rst
    
  def t(self):
    rst = self.T.copy()
    rst=self.setGradFn(rst,"T")
    return rst
    
  def rot180(self):
    a = self.reshape(self.size)
    a = a[::-1]
    a = a.reshape(self.shape)
    return a
  def onehot(self,n=None):
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
     
  def pad(self,pad_width,mode="constant"):
    return np.pad(self,pad_width,mode=mode)
  
  def kron(self,b):
    return np.kron(self,b)

  def clip(self,m,n): return np.clip(self,m,n)
  
  def hsplit(self,m): return np.split(self,m,1)
  def vsplit(self,m): return np.split(self,m,0)
  def split(self,m,axis=1): return np.split(self,m,axis)

  def take(self,axis,p): return np.take(self,axis,p)
        
  def value(self): return self.tolist()
  
  def setGrad(self,bool=True):
    self.requiresGrad=bool
    self.grad=None
    self.gradFn=None
    if (bool and self.isLeaf):
      self.gradFn=autograd.Variable(self)
  
  @property
  def isLeaf(self):
    return (not isinstance(self.gradFn,autograd.Operate))

  def expression(self): return self.gradFn and self.gradFn.expression()
  def gradExpression(self):
    if (not self.gradFn): return None
    return self.gradFn.gradExpression()
  def backward(self,prevOp=None):
    if (not self.requiresGrad): raise Exception("after call setGrad(true) ,then use this function")
    if (prevOp): prevOp = autograd.Constant(prevOp)
    variables=self.gradFn.variables()
    for v in variables:
      op=self.gradFn.backward(prevOp,v)
      a=op.eval()
      v.data.grad = v.data.grad.add(a) if isinstance(v.data.grad,Vector) else a
  def gradClear(self):
    self.gradFn.clearData()
    self.grad=None

class Dim(object):
  def __init__(self):
    self.dtype='float32'
    self.Vector=Vector
    self.random = np.random
    self.nn = NN(self)
    self.optim = Optimizer()
  def vector(self,*a,dtype='float32'):
    rst=[]
    #if isinstance(a[0],tuple): a=a[0]
    #if isinstance(a[0],list): a=a[0]
    for x in a:
      if isinstance(x,Vector): 
        rst.append(x)
      elif isinstance(x,int) or isinstance(x,float): 
        x=np.array([x],dtype)
        rst.append(Vector(shape=x.shape,buffer=x,dtype=x.dtype.name))
      elif isinstance(x,list):
        x=np.array(x,dtype)
        rst.append(Vector(shape=x.shape,buffer=x,dtype=x.dtype.name))
      else:
        rst.append(Vector(shape=x.shape,buffer=x,dtype=x.dtype.name))
    if len(rst)==1: rst=rst[0]
    return rst
    
  def empty(self,shape):
    return self.vector(np.empty(shape))  
  def fill(self,n,shape):
    a=self.empty(shape)
    a.fill(n)
    return a  
  def array(self,a,dtype='float32'):
    return self.vector(a,dtype=dtype)  
  def flatten(self,a):
    return self.vector(np.flatten(a))
  def copy(self,a):
    return self.vector(np.copy(a)) 
  def save(self,a,file): return np.save(file)
  def load(self,file): return np.load(file)
  def arange(self,start,end=None,step=1,dtype="float32"):
    if (end==None):
      end=start
      start=0
    return self.vector(np.arange(start,end,step,dtype))
  def mat(self,str_mat,dtype="float32"):
    return self.vector(self.vector(np.mat(str_mat,dtype)))
  def zeros(self,shape,dtype='float32'):
    return self.vector(np.zeros(shape,dtype))
  def ones(self,shape,dtype='float32'):
    return self.vector(np.ones(shape,dtype))
  def eye(self,number,dtype='float32'):
    return self.vector(np.eye(number,dtype=dtype))
  def diag(self,a,dtype='float32'):
    return self.vector(np.diag(a,dtype=dtype))
        
  def reshape(self,a,*d):
    if (type(d[0])==tuple): d=d[0]
    return self.vector(np.reshape(a,d))
  def swapaxes(self,a,m,n): return self.vector(np.swapaxes(a,m,n))
  def squeeze(self,a):pass
  
  def poly1d(self,a): return np.poly1d(a)
  def polyadd(self,p1,p2): return p1.add(p2)
  def polysub(self,p1,p2): return p1.sub(p2)
  def polymul(self,p1,p2): return p1.mul(p2)
  def polydiv(self,p1,p2): return p1.div(p2)
  def polyval(self,p,a)  : return p.val(a)
  
  def rand(self,shape): return self.vector(np.random.random(shape))
  def randint(self,start,end,shape):return self.vector(np.random.randint(start,end,shape))
  def randn(self,*shape): return self.vector(np.random.randn(*shape))
  
  def radians(self,a): a=self.vector(a);return a.radians() 
  def sin(self,a): a=self.vector(a);return a.sin()
  def cos(self,a): a=self.vector(a);return a.cos()
  def tan(self,a): a=self.vector(a);return a.tan()
  def asin(self,a): a=self.vector(a);return a.asin()
  def acos(self,a): a=self.vector(a);return a.acos()
  def atan(self,a): a=self.vector(a);return a.atan()
  def asinh(self,a): a=self.vector(a);return a.asinh()
  def acosh(self,a): a=self.vector(a);return a.acosh()
  def atanh(self,a): a=self.vector(a);return a.atanh()
  def sinh(self,a): a=self.vector(a);return a.sinh()
  def cosh(self,a): a=self.vector(a);return a.cosh()
  def tanh(self,a): a=self.vector(a);return a.tanh()

  def log(self,a): a=self.vector(a);return a.log()
  def log2(self,a): a=self.vector(a);return a.log2()
  def log10(self,a): a=self.vector(a);return a.log10()
  def exp(self,a): a=self.vector(a);return a.exp()
  def sqrt(self,a): a=self.vector(a);return a.sqrt()
  def square(self,a): a=self.vector(a);return a.square()
  def pow(self,a,n): a=self.vector(a);return a.pow(n)
  def floor(self,a): a=self.vector(a);return a.floor()
  def ceil(self,a): a=self.vector(a);return a.ceil()
  def around(self,a,n): a=self.vector(a);return a.around(n)
  def abs(self,a): a=self.vector(a);return a.abs()
  def neg(self,a): a=self.vector(a);return a.neg()
  def reciprocal(self,a): a=self.vector(a);return a.reciprocal()

  def add(self,a,b): a,b=self.vector(a,b);return a.add(b)
  def sub(self,a,b): a,b=self.vector(a,b);return a.sub(b)
  def mul(self,a,b): a,b=self.vector(a,b);return a.mul(b)
  def div(self,a,b): a,b=self.vector(a,b);return a.div(b)

  def mod(self,a,b): a,b=self.vector(a,b);return a.mod(b)
  def subtract(self,a,b): return self.sub(a,b)
  def multiply(self,a,b): return self.mul(a,b)
  def divide(self,a,b):   return self.div(a,b)
  def negative(slef,a): return self.neg(a)
  def power(self,a,n):  return self.pow(a,n)
  
  def sign(self,a): a=self.vector(a);return a.sign()
  def gt(self,a,b): a,b=self.vector(a,b);return a.gt(b)
  def lt(self,a,b): a,b=self.vector(a,b);return a.lt(b)
  def gt(self,a,b): a,b=self.vector(a,b);return a.ge(b)
  def lt(self,a,b): a,b=self.vector(a,b);return a.le(b)
  def eq(self,a,b): a,b=self.vector(a,b);return a.eq(b)
  def ne(self,a,b): a,b=self.vector(a,b);return a.ne(b)
  def allclose(self,a,b): a,b=self.vector(a,b);return a.allclose(b)
  def all(self,a,axis=None): a=self.vector(a);return a.all(axis)
  def any(self,a,axis=None): a=self.vector(a);return a.any(axis)

  def sum(self,a,axis=None): a=self.vector(a);return a.sum(axis)
  def mean(self,a,axis=None): a=self.vector(a);return a.mean(axis)
  def max(self,a,axis=None):  a=self.vector(a);return a.max(axis)
  def min(self,a,axis=None):  a=self.vector(a);return a.min(axis)
  def argmax(self,a,axis=None): a=self.vector(a);return a.argmax(axis)
  def argmin(self,a,axis=None): a=self.vector(a);return a.argmin(axis)
  def var(self,a,axis=None): a=self.vector(a);return a.var(axis)
  def std(self,a,axis=None): a=self.vector(a);return a.std(axis)
  def cov(self,a,axis=None): a=self.vector(a);return a.cov(axis)
  def ptp(self,a,axis=None): a=self.vector(a);return a.ptp(axis)
  def median(self,a,axis=None): a=self.vector(a);return a.median(axis)
  
  def sort(self,a,axis=None): a=self.vector(a);return a.sort(axis)
  
  #normal(a,N){a=this.ensureVector(a);return a.normal(N)}
  #minmaxNormal(a){a=this.ensureVector(a);return a.minmaxNormal()}
  

  def dot(self,a,b): a,b=self.vector(a,b);return a.dot(b)
  def matmul(self,a,b): return self.dot(a,b)
  def trace(self,a): a=self.vector(a);return a.trace()

  def pad(self,a,pad_width,mode="constant"):a=self.vector(a);return a.pad(pad_width,mode)

  def onehot(a,n): a=self.vector(a);return a.onehot(n)
  def kron(self,a,b): a,b=self.vector(a,b);return a.kron(b)
  def clip(self,a,m,n): a=self.vector(a);return a.clip(m,n)
  def hstack(self,a): return self.vector(np.hstack(a))
  def vstack(self,a): return self.vector(np.vstack(a))
  def stack(self,a,axis=1): return self.vector(np.stack(a,axis))
  
  def concat(self,a,axis=1): return self.vector(np.concatenate(a,axis))
  
  def hsplit(self,a,m): a=self.vector(a);return a.split(m,1)
  def vsplit(self,a,m): a=self.vector(a);return a.split(m,0)
  def split(self,a,m,axis=1): a=self.vector(a);return a.split(m,axis)

  def take(self,a,axis,p): a=self.vector(a);return a.take(axis,p)
  
  '''
  where(){}
  nonzero(){}
  
  fftConv(a,b){
    if (!Array.isArray(a) || !Array.isArray(b)) throw new Error(`a、b参数必须都是数组`)
    let n = a.length + b.length -1 
    let N = 2**(parseInt(Math.log2(n))+1)
    let numa=N-a.length
    let numb=N-b.length
    for(let i=0;i<numa;i++) a.unshift(0)
    for(let i=0;i<numb;i++) b.unshift(0)
    let A=this.array(this.fft.fft(a))
    let B=this.array(this.fft.fft(b))
    let C=A.mul(B)
    return this.fft.ifft(C.data)
  }
  '''