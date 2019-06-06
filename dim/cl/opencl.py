import dim
import pyopencl as cl
import pyopencl.clmath
from pyopencl.clrandom import rand
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool

import os
os.environ['PYOPENCL_CTX'] = '0:1'
os.environ['PYOPENCL_COMPILER_OUTPUT']='1'

class Opencl:
  def __init__(self,p=0,d=1):
    #export PYOPENCL_CTX='0:1'
    self.device_type = cl.device_type
    self.mf = cl.mem_flags
    self.Buffer = cl.Buffer
    self.clarray = cl.array
    self.clmath= cl.clmath
    self.clrand = cl.clrandom.rand
    self.Array = cl.array.Array
    self.Program = cl.Program
    self.Kernel = cl.Kernel
    self.enqueue_nd_range_kernel=cl.enqueue_nd_range_kernel
    self.pool = ThreadPool(8)
    try:
      self.platforms = cl.get_platforms()
      self.devices=self.platforms[p].get_devices()
      self.ctx = cl.Context(devices=[self.devices[d]])
      #self.ctx=cl.create_some_context()
      self.queue=cl.CommandQueue(self.ctx)
    except Exception as e:
      print("Opencl is not available.",e)
      self.platforms=None
      self.ctx=None
      self.queue=None
  def choose_device(self,n=1):
    self.ctx = cl.Context(devices=[self.devices[n]])
    self.queue=cl.CommandQueue(self.ctx)
  def is_available(self):
    return self.ctx is not None
  def device_count(self):
    return len(self.devices)
  def to_device(self,obj):
    #return self.array.to_device(self.queue,obj)
    if not isinstance(obj,dim.Vector):
      raise Exception("arg is not instance of dim.Vector")

    #buf = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=obj)
    buf = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=obj)
    rst= Array(self.queue,obj.shape,str(obj.dtype),base_data=buf)
    if obj.requiresGrad:
      rst.setGrad()
    return rst
  def to_host(self,obj):
    if not isinstance(obj,Array):
      raise Exception("arg is not instance of opencl.Array")
    return obj.get()
  def array(self,a):
    if isinstance(a,self.Array):
      return  Array(a.queue,a.shape,a.dtype,base_data=a.data)   
    return dim.array(a).to_device()
  def rand(self,shape,dtype='float32',a=0,b=1):
    return  self.array(self.clrand(self.queue, shape, dtype=dtype, a=a, b=b))
  def zeros(self,shape,dtype='float32'):
    return self.array(self.clarray.zeros(self.queue,shape,dtype=dtype))
  def ones(self,shape,dtype='float32'):
    return self.array(self.clarray.empty(self.queue,shape,dtype=dtype)).fill(1)
  def empty(self,shape,dtype='float32'):
    return self.clarray.empty(self.queue,shape,dtype=dtype)
  def init(self):
    with open("test.cl") as fs:
      source=fs.read()
    self.program=self.Program(self.ctx,source).build()
  def mm(self,a,b):
    #kernel = self.Kernel(self.program, "mm")
    #self.program.mult(self.queue, a.shape, None, a.data, np.float32(2), np.int32(3))
    shape=(a.shape[0],b.shape[1])
    out=self.zeros(shape)
    width1 = np.int32(a.shape[1])
    width2 = np.int32(b.shape[1])
    event=self.program.mm(self.queue,shape,None,
        a.data,b.data,out.data,width1,width2)
    event.wait()
    return out
    #a_result = dim.empty(a.shape)
    #print(a,a_result,a_result.shape)
    #cl.enqueue_copy(self.queue, a.data, a_result).wait()     
  def conv2d(self,inputs,filters,stride=1,padding=0):
    '''
    if (len(inputs.shape)!=4):
      raise Exception("input({})不符合[miniBatch*inChannels*H*W]的形状要求".format(inputs.shape)) 
    if (len(filters.shape)!=4):
      raise Exception("filter({})不符合[outChannels*inChannels*H*W]的形状要求".format(filters.shape)) 
    if (inputs.shape[1]!=filters.shape[1]):
      raise Exception("input({})与filter({})中channels数不一致".format(inputs.shape,filters.shape))
    '''
    h1=inputs.shape[0]
    w1=inputs.shape[1]
    h2=filters.shape[0]
    w2=filters.shape[1]
    h3= int((h1-h2-2*padding)/stride+1)
    w3= int((w1-w2-2*padding)/stride+1)
    shape=(h3,w3)
    out=self.zeros(shape)
    width1 = np.int32(out.shape[1])
    width2 = np.int32(w1)
    hf = np.int32(filters.shape[0])
    wf = np.int32(filters.shape[1])
    event=self.program.conv2d(self.queue,shape,None,
        inputs.data,filters.data,out.data,width1,width2,hf,wf)
    event.wait()
    return out

class Array(cl.array.Array):
  def __init__(
        self, queue, shape, dtype, strides=None, offset=0,
        allocator=None, base_data=None):
    cl.array.Array.__init__(
        self, queue, shape, dtype, strides=strides, allocator=allocator,
        data=base_data, offset=offset)
    self.__gradFn=None
    self.__grad=None
    self.__requiresGrad=False
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
    if (isinstance(left,Array) and left.requiresGrad) or (isinstance(right,Array) and right.requiresGrad):
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
  @property
  def T(self):
    rst=self.transpose()
    return dim.cl.array(rst)
  def __repr__(self):
    return cl.array.Array.__repr__(self)+"\nrun on device:"+str(self.queue.device)
  def __add__(self,other):
    rst = cl.array.Array.__add__(self,other)
    rst = self.setGradFn(rst,"add",right=other)
    return rst
  def __mul__(self,other):
    rst = cl.array.Array.__mul__(self,other)
    rst = self.setGradFn(rst,"mul",right=other)
    return rst
  def add(self,other):
    return self+other
  def mm(self,other):
    rst = dim.cl.mm(self,other)
    rst = self.setGradFn(rst,"dot",right=other)
    return rst
  def dot(self,other):
    return self.mm(other)
  def __getitem1__(self,index):
    rst = cl.array.Array.__getitem__(self,index)
    return rst
    #return dim.cl.array(rst)
  def mm1(self,other):
    M=self.shape[0]
    P=self.shape[1]
    Q=other.shape[0]
    N=other.shape[1]
    if P!=Q: raise Exception("shape not right")
    c=[0]*M*N
    other1=other.reshape((N,Q))
    for i in range(M):
      for j in range(N):
        c[i*j+j] =cl.array.dot(self[i],other1[j])
    return c
  def conv2d(self,filters,stride=1,padding=0):
    rst = dim.cl.conv2d(self,filters,stride,padding)
    rst = self.setGradFn(rst,"conv2d",right=filters)
    return rst
  def get(self):
    return dim.vector(cl.array.Array.get(self))
  def view(self):
    return dim.cl.array(cl.array.Array.view(self))
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
      v.data.grad = v.data.grad+grad if v.data.grad is not None else grad
  def gradClear(self):
    self.gradFn.clearData()
    self.grad=None
 
  def hello(self):
    print("hello")