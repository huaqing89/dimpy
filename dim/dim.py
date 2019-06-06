import math
import numpy as np
from .vector import Vector
#from nn import NN,Optimizer
  
def vector(*a,dtype='float32'):
  rst=[]
  #if isinstance(a[0],tuple): a=a[0]
  #if isinstance(a[0],list): a=a[0]
  for x in a:
    if isinstance(x,Vector): 
      rst.append(x)
    elif isinstance(x,(int,float)): 
      x=np.array([x],dtype)
      rst.append(Vector(shape=x.shape,buffer=x,dtype=str(x.dtype)))
    elif isinstance(x,list):
      x=np.array(x,dtype)
      rst.append(Vector(shape=x.shape,buffer=x,dtype=str(x.dtype)))
    else:
      rst.append(Vector(shape=x.shape,buffer=x,dtype=str(x.dtype)))
  if len(rst)==1: rst=rst[0]
  return rst
  
def empty(*shape):
  if isinstance(shape[0],(list,tuple)): shape=shape[0]
  return vector(np.empty(shape))

def fill(n,*shape):
  a=empty(*shape)
  a.fill(n)
  return a  

def array(a,dtype='float32'):
  return vector(a,dtype=dtype)  

def flatten(a):
  return vector(np.flatten(a))
def copy(a):
  return vector(np.copy(a)) 
def save(a,file): return np.save(file)
def load(file): return np.load(file)
def arange(start,end=None,step=1,dtype="float32"):
  if (end==None):
    end=start
    start=0
  return vector(np.arange(start,end,step,dtype))
def mat(str_mat,dtype="float32"):
  return vector(vector(np.mat(str_mat,dtype)))
def zeros(shape,dtype='float32'):
  return vector(np.zeros(shape,dtype))
def ones(shape,dtype='float32'):
  return vector(np.ones(shape,dtype))
def eye(number,dtype='float32'):
  return vector(np.eye(number,dtype=dtype))
def diag(a,dtype='float32'):
  return vector(np.diag(a,dtype=dtype))
      
def reshape(a,*d):
  if (type(d[0])==tuple): d=d[0]
  return vector(np.reshape(a,d))
def swapaxes(a,m,n): return vector(np.swapaxes(a,m,n))
def squeeze(a):pass

def poly1d(a): return np.poly1d(a)
def polyadd(p1,p2): return p1.add(p2)
def polysub(p1,p2): return p1.sub(p2)
def polymul(p1,p2): return p1.mul(p2)
def polydiv(p1,p2): return p1.div(p2)
def polyval(p,a)  : return p.val(a)

def random(start,end,*shape): 
  if isinstance(shape[0],(list,tuple)): shape=shape[0]
  return vector(((end-start)*np.random.random(shape).astype('float32').round(4)+start))
def randint(start,end,*shape):
  if isinstance(shape[0],(list,tuple)): shape=shape[0]
  return vector(np.random.randint(start,end,shape).astype('int32'))
def randn(*shape): return vector(np.random.randn(*shape).astype('float32').round(4))
def uniform(start,end,*shape):
  if isinstance(shape[0],(list,tuple)): shape=shape[0]
  return vector(np.random.uniform(start,end,shape).astype('float32').round(4))
def normal(mean=0,std=1,*shape):
  if isinstance(shape[0],(list,tuple)): shape=shape[0]
  return randn(*shape)*std+mean
#minmaxNormal(a){a=this.ensureVector(a);return a.minmaxNormal()}
def shuffle(a):
  np.random.shuffle(a)
  return a

def radians(a): a=vector(a);return a.radians() 
def sin(a): a=vector(a);return a.sin()
def cos(a): a=vector(a);return a.cos()
def tan(a): a=vector(a);return a.tan()
def asin(a): a=vector(a);return a.asin()
def acos(a): a=vector(a);return a.acos()
def atan(a): a=vector(a);return a.atan()
def asinh(a): a=vector(a);return a.asinh()
def acosh(a): a=vector(a);return a.acosh()
def atanh(a): a=vector(a);return a.atanh()
def sinh(a): a=vector(a);return a.sinh()
def cosh(a): a=vector(a);return a.cosh()
def tanh(a): a=vector(a);return a.tanh()

def log(a): a=vector(a);return a.log()
def log2(a): a=vector(a);return a.log2()
def log10(a): a=vector(a);return a.log10()
def exp(a): a=vector(a);return a.exp()
def sqrt(a): a=vector(a);return a.sqrt()
def square(a): a=vector(a);return a.square()
def pow(a,n): a=vector(a);return a.pow(n)
def floor(a): a=vector(a);return a.floor()
def ceil(a): a=vector(a);return a.ceil()
def around(a,n): a=vector(a);return a.around(n)
def abs(a): a=vector(a);return a.abs()
def neg(a): a=vector(a);return a.neg()
def reciprocal(a): a=vector(a);return a.reciprocal()

def add(a,b): a,b=vector(a,b);return a.add(b)
def sub(a,b): a,b=vector(a,b);return a.sub(b)
def mul(a,b): a,b=vector(a,b);return a.mul(b)
def div(a,b): a,b=vector(a,b);return a.div(b)

def mod(a,b): a,b=vector(a,b);return a.mod(b)
def subtract(a,b): return sub(a,b)
def multiply(a,b): return mul(a,b)
def divide(a,b):   return div(a,b)
def negative(slef,a): return neg(a)
def power(a,n):  return pow(a,n)

def sign(a): a=vector(a);return a.sign()
def gt(a,b): a,b=vector(a,b);return a.gt(b)
def lt(a,b): a,b=vector(a,b);return a.lt(b)
def gt(a,b): a,b=vector(a,b);return a.ge(b)
def lt(a,b): a,b=vector(a,b);return a.le(b)
def eq(a,b): a,b=vector(a,b);return a.eq(b)
def ne(a,b): a,b=vector(a,b);return a.ne(b)
def allclose(a,b): a,b=vector(a,b);return a.allclose(b)
def all(a,axis=None): a=vector(a);return a.all(axis)
def any(a,axis=None): a=vector(a);return a.any(axis)

def sum(a,axis=None,**kwargs): a=vector(a);return a.sum(axis,**kwargs)
def mean(a,axis=None,**kwargs): a=vector(a);return a.mean(axis,**kwargs)
def max(a,axis=None,**kwargs):  a=vector(a);return a.max(axis,**kwargs)
def min(a,axis=None,**kwargs):  a=vector(a);return a.min(axis,**kwargs)
def argmax(a,axis=None,**kwargs): a=vector(a);return a.argmax(axis,**kwargs)
def argmin(a,axis=None,**kwargs): a=vector(a);return a.argmin(axis,**kwargs)
def var(a,axis=None,**kwargs): a=vector(a);return a.var(axis,**kwargs)
def std(a,axis=None,**kwargs): a=vector(a);return a.std(axis,**kwargs)
def cov(a,axis=None,**kwargs): a=vector(a);return a.cov(axis,**kwargs)
def ptp(a,axis=None,**kwargs): a=vector(a);return a.ptp(axis,**kwargs)
def median(a,axis=None,**kwargs): a=vector(a);return a.median(axis,**kwargs)
  
def sort(a,axis=None,**kwargs): a=vector(a);return a.sort(axis,**kwargs)


def dot(a,b): a,b=vector(a,b);return a.dot(b)
def matmul(a,b): return dot(a,b)
def trace(a): a=vector(a);return a.trace()

def pad(a,pad_width,mode="constant"):a=vector(a);return a.pad(pad_width,mode)

def onehotEncode(a,n): a=vector(a);return a.onehotEncode(n)
def labelEncode(a,n): a=vector(a);return a.labelEncode(n)
def kron(a,b): a,b=vector(a,b);return a.kron(b)
def clip(a,m,n): a=vector(a);return a.clip(m,n)
def hstack(a): return vector(np.hstack(a))
def vstack(a): return vector(np.vstack(a))
def stack(a,axis=1): return vector(np.stack(a,axis))

def concat(a,axis=1): return vector(np.concatenate(a,axis))

def hsplit(a,m): a=vector(a);return a.split(m,1)
def vsplit(a,m): a=vector(a);return a.split(m,0)
def split(a,m,axis=0): a=vector(a);return a.split(m,axis)

def take(a,axis,p): a=vector(a);return a.take(axis,p)

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