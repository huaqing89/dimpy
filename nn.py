#coding:utf-8
import autograd
import math

class Functional(object):
  def __init__(self,dim):
     self.dim = dim
     self.grad = autograd
     self.Vector = dim.Vector

  def softmax(self,x,axis=1):
    x=self.dim.vector(x)
    y=x.exp()
    sum=y.sum(axis)
    rst=y/sum
    if (axis==1):
      rst = x.setGradFn(rst,"softmax")    
    return rst

  def softmaxDeri(self,x,a):
    x,a=self.dim.vector(x,a)
    argmax=a.argmax(1).value()
    data=x.value
    '''rst = self.Vector(data.map((y,i)=>{
      return y.map((z,j)=>argmax[i]==j?z*(1-z):-z*y[argmax[i]])
    }))
    '''
    return rst
  #Activation Function
  def relu(self,x):
    rst=x.copy()
    rst[x<0]=0
    rst = x.setGradFn(rst,"relu")
    return rst

  def reluDeri(self,x):
    rst= x.copy()
    rst[x>0]=1
    rst[x<0]=0
    return rst

  def relu6(self,x):
    rst=x.copy()
    rst[x>6]=6
    rst[x<0]=0
    rst = x.setGradFn(rst,"relu6")
    return rst
  
  def relu6Deri(self,x):
    rst = x.copy()
    rst[x>6]=1
    rst[x<=6]=0
    return rst
  
  def softplus(self,x):
    x=self.dim.vector(x)
    return (x.exp()+1).log()
    
  def sigmoid(self,x):
    x=self.dim.vector(x)
    return 1/(1+(-x).exp())
  
  def tanh(self,x):
    x=self.dim.vector(x)
    return x.tanh()
    
  def dropout(a,keep):
    if (keep<=0 or keep>1): raise Exception("keep_prob参数必须属于(0,1]")
    a=self.dim.vector(a)
    arr=[]
    '''return new self.Vector(a.data.map((x,i)=>{
      if (x instanceof self.Vector) return self.dropout(x,keep)
      if (i==0){
        let remain=a.data.length*keep
        for (let j=0;j<a.data.length;j++) arr.append(j)
        arr = self.random.shuffle(arr).slice(0,remain)
      }
      return (arr.indexOf(i)>=0)?x/keep:0
    }))
    '''

  #Loss Function
  def mseLoss(self,a,y):
    #also named L2
    a,y=self.dim.vector(a,y)
    return y.sub(a).square().mean()
  
  def binaryCrossEntropy(self,a,y):
    a,y=self.dim.vector(a,y)
    return (y*a.log() + (1-y)*(1-a).log()).sum()
  
  def crossEntropy(self,a,y):
    a,y=self.dim.vector(a,y)
    b=self.softmax(a,1)
    y_onehot=y.onehot(b.shape[1])

    rst = y_onehot.mul(b.log()).sum(1).neg().mean()
    rst = a.setGradFn(rst,"crossEntropy",right=y)
    return rst

  def crossEntropyDeri(self,a,y):
    a,y=self.dim.vector(a,y)
    b=self.softmax(a,1)
    y_onehot=self.dim.onehot(y,a.shape[1])
    rst = b.sub(y_onehot).div(b.shape[0])
    return rst
  
  def logcoshLoss(self,a,y):
    a,y=self.dim.vector(a,y)
    return y.sub(a).cosh().log().sum()

  #cnn function
  def conv1d(self,inputs, filters, stride=1, padding=0):
    if (len(inputs.shape)!=3):
      raise Exception("input({})不符合[miniBatch*inChannels*W]的形状要求".format(inputs.shape))
    if (len(filters.shape)!=3):
      raise Exception("filter({})不符合[outChannels*inChannels*W]的形状要求".format(filters.shape))
    if (inputs.shape[1]!=filters.shape[1]): 
      raise Exception("input({})与filter({})中channels数不一致".format(inputs.shape,filters.shape))
    a=[]
    for i in range(inputs.shape[0]): #miniBatch
      a.append([])
      for j in range(filters.shape[0]): #outChannel
        a[i].append([])
        for k in range(filters.shape[1]): #inChannel
           bat=inputs[i,k].pad(padding)
           kernel = filters[j,k]
           iw=bat.size
           fw=kernel.size
           w=math.floor((iw-fw)/stride+1)
           for l in range(w):
             value = bat[l*stride:l*stride+fw].dot(kernel)
             if len(a[i][j])<=l:
               a[i][j].append(value)
             else:
               a[i][j][l]+=value
    rst=self.dim.vector(a)
    #P1=(((In-1)*S+F) -(((In-F+2*P0)/S)+1))/2let In=input.shape[2]
    In=inputs.shape[2]
    F=filters.shape[2]
    S=stride
    P0=padding
    gradPadding=math.floor(((((In-1)*S+F) -(((In-F+2*P0)/S)+1))/2))
    rst = inputs.setGradFn(rst,"conv1d",left=inputs,right=filters,args={"padding":gradPadding})
    return rst

  def conv2d(self,inputs, filters, stride=1, padding=0):
    if (len(inputs.shape)!=4):
      raise Exception("input({})不符合[miniBatch*inChannels*H*W]的形状要求".format(inputs.shape)) 
    if (len(filters.shape)!=4):
      raise Exception("filter({})不符合[outChannels*inChannels*H*W]的形状要求".format(filters.shape)) 
    if (inputs.shape[1]!=filters.shape[1]):
      raise Exception("input({})与filter({})中channels数不一致".format(inputs.shape,filters.shape))
    a=[]
    for i in range(inputs.shape[0]):
      a.append([])
      for j in range(filters.shape[0]):
        a[i].append([])
        for k in range(filters.shape[1]):
          bat = inputs[i,k].pad(padding)
          kernel = filters[j,k]
          ih=bat.shape[0]
          iw=bat.shape[1]
          fh=kernel.shape[0]
          fw=kernel.shape[1]
          w=math.floor((iw-fw)/stride+1)
          h=math.floor((ih-fh)/stride+1)
          for l in range(h):
            if len(a[i][j])<=l:
              a[i][j].append([])
            for m in range(w):
                value=(bat[l*stride:l*stride+fh,m*stride:m*stride+fw]*(kernel)).sum().value()
                if len(a[i][j][l])<=m:
                  a[i][j][l].append(value)
                else:
                  a[i][j][l][m]+=value
    print(a)              
    rst = self.dim.vector(a)
    #P1=(((In-1)*S+F) -(((In-F+2*P0)/S)+1))/2let In=input.shape[2]
    In=inputs.shape[2]
    F=filters.shape[2]
    S=stride
    P0=padding
    gradPadding=int((((In-1)*S+F) -(((In-F+2*P0)/S)+1))/2)
    
    rst = inputs.setGradFn(rst,"conv2d",left=inputs,right=filters,args={"padding":gradPadding})
    return rst
  
  def conv3d(self,inputs, filters, stride=1, padding=0):
    if (len(inputs.shape)!=5):
      raise Exception("input({})不符合[miniBatch*inChannels*D*H*W]的形状要求".format(inputs.shape)) 
    if (len(filters.shape)!=5):
      raise Exception("filter({})不符合[outChannels*inChannels*D*H*W]的形状要求".format(filters.shape)) 
    if (inputs.shape[1]!=filters.shape[1]): 
      raise Exception("input({})与filter({})中channels数不一致".foramt(inputs.shape,filters.shape))

    a=[]
    for i in range(inputs.shape[0]):
      a.append([])
      for j in range(filters.shape[0]):
        a[i].append([])
        for k in range(filters.shape[1]):
          bat = inputs[i,k].pad(padding)
          kernel = filters[j,k]
          ideep=bat.shape[0]
          ih=bat.shape[1]
          iw=bat.shape[2]
          fdeep=kernel.shape[0]
          fh=kernel.shape[1]
          fw=kernel.shape[2]
          d=(ideep-fdeep)/stride+1
          h=(ih-fh)/stride+1
          w=(iw-fw)/stride+1
          for l in range(d):
            if len(a[i][j])<=l: a[i][j].push([])
            for m in range(h):
              if len(a[i][j][l])<=m: a[i][j][l].push([])
              for n in range(w):
                value = (bat[l*stride:l*stride+fdeep,m*stride:m*stride+fh,n*stride:n*stride+fw]*(kernel)).sum().value()
                if len(a[i][j][l][m])<=n:
                  a[i][j][l][m].push(value)
                else:
                  a[i][j][l][m][n]+=value
    return self.dim.vector(a)
         
  def convTranspose1d(self,inputs, filters, stride=1, padding=0):
    if (len(inputs.shape)!=3):
      raise Exception("input({})不符合[miniBatch*inChannels*W]的形状要求".format(inputs.shape)) 
    if (len(filters.shape)!=3):
      raise Exception("filter({})不符合[inChannels*outChannels*W]的形状要求".format(filters.shape)) 
    if (inputs.shape[1]!=filters.shape[0]): 
      raise Exception("input({})与filter({})中channels数不一致".format(inputs.shape,filters.shape))
    #change channel
    filters = self.dim.swapaxes(filters,0,1)
    a=[]
    for i in range(inputs.shape[0]): #miniBatch
      a.append([])
      for j in range(filters.shape[0]): #outChannel
        a[i].append([])
        for k in range(filters.shape[1]): #kernel
          bat = inputs[i,k].pad(padding)
          kernel = filters[j,k].rot180()
          iw=bat.size
          fw=kernel.size
          w=math.floor((iw-fw)/stride+1)
          for l in range(w):
            value = (bat[l*stride:l*stride+fw]*(kernel)).sum()
            if len(a[i][j])<=l:
              a[i][j].append(value)
            else:
              a[i][j][l]+=value
    rst = self.dim.vector(a)
    return rst
  
  def convTranspose2d(self,inputs, filters, stride=1, padding=0):
    #要实现还原运算，padding=((Out-1)*stride-Input+Filter)/2
    if (len(inputs.shape)!=4):
      raise Exception("input({})不符合[miniBatch*inChannels*H*W]的形状要求".format(inputs.shape)) 
    if (len(filters.shape)!=4):
      raise Exception("filter({})不符合[outChannels*inChannels*H*W]的形状要求".format(filters.shape)) 
    if (inputs.shape[1]!=filters.shape[0]):
      raise Exception("input({})与filter({})中channels数不一致".format(inputs.shape,filters.shape))
    filters = self.dim.swapaxes(filters,0,1)
    a=[]
    for i in range(inputs.shape[0]):
      a.append([])
      for j in range(filters.shape[1]):
         a[i].append([])
         for k in range(filters.shape[2]):
            bat = inputs[i,k].pad(padding)
            kernel = filters[j,k].rot180()
            ih=bat.shape[0]
            iw=bat.shape[1]
            fh=kernel.shape[0]
            fw=kernel.shape[1]
            w=math.floor((iw-fw)/stride+1)
            h=math.floor((ih-fh)/stride+1)
            for l in range(h):
              if len(a[i][j])<=l: a[i][j].append([])
              for m in range(w):
                value =(bat[l*stride:l*stride+fh,m*stride:m*stride+fw]*(kernel)).sum().value()
                if len(a[i][j][l])<=m:
                  a[i][j][l].append(value)
                else:
                  a[i][j][l][m]+=value
    return self.dim.vector(a)
  
  def convTranspose3d(self,inputs, filters, stride=1, padding=0):
    #要实现还原运算，padding=((Out-1)*stride-Input+Filter)/2
    if (len(inputs.shape)!=5):
      raise Exception("input({})不符合[miniBatch*inChannels*D*H*W]的形状要求".format(inputs.shape)) 
    if (len(filters.shape)!=5):
      raise Exception("filter({})不符合[outChannels*inChannels*D*H*W]的形状要求".format(filters.shape)) 
    if (input.shape[1]!=filter.shape[0]):
      raise Exception("input({})与filter({})中channels数不一致".format(inputs.shape,filters.shape))
    filters = dim.swapaxes(filters,0,1)
    a=[]
    for i in range(inputs.shape[0]):
      a.append([])
      for j in range(filters.shape[1]):
         a[i].append([])
         for k in range(filters.shape[2]):
            bat = inputs[i,k].pad(padding)
            kernel = filters[j,k].rot180()
            ideep=bat.shape[0]
            ih=bat.shape[1]
            iw=bat.shape[2]
            fdeep=kernel.shape[0]
            fh=kernel.shape[1]
            fw=kernel.shape[2]
            d=math.floor((ideep-fdeep)/stride+1)
            w=math.floor((iw-fw)/stride+1)
            h=math.floor((ih-fh)/stride+1)
            for l in range(d):
              if len(a[i][j])<=l: a[i][j].append([])
              for m in range(h):
                if len(a[i][j][l])<=m: a[i][j][l].append([])
                for n in range(w):
                  value =(bat[l*stride:l*stride+fdeep,m*stride:m*stride+fh,n*stride:n*stride+fw]*(kernel)).sum().value()
                  if len(a[i][j][l][m])<=n:
                    a[i][j][l][m].append(value)
                  else:
                    a[i][j][l][m][n]+=value
    return self.dim.vector(a)
  
  #Pool Function
  def maxPool1d(self,inputs,ks,indices=[],padding=0):
    if (len(inputs.shape)!=3):
       raise Exception("input({})不符合[miniBatch*inChannels*W]的形状要求".format(inputs.shape)) 
    
    ks=int(ks)
    a=[]
    
    for i,channel in enumerate(inputs):
      a.append([])
      indices.append([])
      for j,kernel in enumerate(channel):
        a[i].append([])
        indices[i].append([])
        kernel=kernel.pad(padding)
        iw=kernel.size
        fw=ks
        w=math.floor((iw-fw)/ks+1)
        for k in range(w):
          flip=kernel[k*ks:k*ks+fw]
          a[i][j].append(flip.max())
          indices[i][j].append(flip.argmax())
    print("maxPool1d a:",a)
    print("maxPool1d indices:",indices)
    rst = self.dim.vector(a)
    rst = inputs.setGradFn(rst,"maxPool1d",left=inputs,right=ks,args={"indices":indices})
    return rst
  def avgPool1d(self,inputs,ks,padding=0):
    if (len(inputs.shape)!=3):
       raise Exception("input({})不符合[miniBatch*inChannels*W]的形状要求".format(inputs.shape)) 
    
    ks=int(ks)
    a=[]
    
    for i,channel in enumerate(inputs):
      a.append([])
      for j,kernel in enumerate(channel):
        a[i].append([])
        kernel=kernel.pad(padding)
        iw=kernel.size
        fw=ks
        w=math.floor((iw-fw)/ks+1)
        for k in range(w):
          flip=kernel[k*ks:k*ks+fw]
          a[i][j].append(flip.mean())
    rst = self.dim.vector(a)
    rst = inputs.setGradFn(rst,"avgPool1d",left=inputs,right=ks)
    return rst
  def maxPool2d(self,inputs,ks,indices=[],padding=0):
    if (len(inputs.shape)!=4):
       raise Exception("input({})不符合[miniBatch*inChannels*H*W]的形状要求".format(inputs.shape)) 
    
    ks=int(ks)
    a=[]
    for i,channel in enumerate(inputs):
      a.append([])
      indices.append([])
      for j,kernel in enumerate(channel):
        a[i].append([])
        indices[i].append([])
        kernel = kernel.pad(padding)
        ih=kernel.shape[0]
        iw=kernel.shape[1]
        fh=ks
        fw=ks
        w=math.floor((iw-fw)/ks+1)
        h=math.floor((ih-fh)/ks+1)
        for k in range(h):
          a[i][j].append([])
          indices[i][j].append([])
          for l in range(w):
            flip=kernel[k*ks:k*ks+fh,l*ks:l*ks+fw]
            a[i][j][k].append(flip.max())
            indices[i][j][k].append(flip.argmax())
    
    rst =  self.dim.vector(a)
    rst =  inputs.setGradFn(rst,"maxPool2d",left=inputs,right=ks,args={"indices":indices})
    return rst
  
  def avgPool2d(self,inputs,ks,padding=0):
    if (len(inputs.shape)!=4):
       raise Exception("input({})不符合[miniBatch*inChannels*H*W]的形状要求".format(inputs.shape)) 
    
    ks=int(ks)
    a=[]
    for i,channel in enumerate(inputs):
      a.append([])
      for j,kernel in enumerate(channel):
        a[i].append([])
        kernel = kernel.pad(padding)
        ih=kernel.shape[0]
        iw=kernel.shape[1]
        fh=ks
        fw=ks
        w=math.floor((iw-fw)/ks+1)
        h=math.floor((ih-fh)/ks+1)
        for k in range(h):
          a[i][j].append([])
          for l in range(w):
            flip=kernel[k*ks:k*ks+fh,l*ks:l*ks+fw]
            a[i][j][k].append(flip.avg())
      
    rst =  self.dim.vector(a)
    rst =  inputs.setGradFn(rst,"avgPool2d",left=inputs,right=ks)
    return rst

  def maxPool3d(self):pass
  def avgPool3d(self):pass
  
  def maxUnpool1d(self,inputs,indices,ks):
    if (len(inputs.shape)!=3):
       raise Exception("input({})不符合[miniBatch*inChannels*W]的形状要求".format(inputs.shape)) 
    print("maxUnpool1d indices:",indices)
    indices=self.dim.vector(indices)
    if (inputs.shape!=indices.shape):
      raise Exception("input({})与indices({})的形状不一致".format(inputs.shape,indices.shape))
    a=[]
    ks=int(ks)
    for i,channel in enumerate(inputs):
      a.append([])
      for j,kernel in enumerate(channel):
        a[i].append([])
        p=[]
        w=kernel.size
        for k in range(w):
          factor = self.dim.zeros(ks)
          r=math.floor(indices[i,j,k]%ks)
          factor[r]=1
          p.append(kernel[k]*factor)
        a[i][j]=self.dim.concat(p,0)
    
    rst = self.dim.vector(a)
    return rst
  def avgUnpool1d(self,inputs,ks):
    if (len(inputs.shape)!=3):
       raise Exception("input({})不符合[miniBatch*inChannels*W]的形状要求".format(inputs.shape)) 
    ks=int(ks)
    a=[]
    factor = self.dim.fill(1/ks,ks)
    for i,channel in enumerate(inputs):
      a.append([])
      for j,kernel in enumerate(channel):
        a[i].append(self.dim.kron(kernel,factor))
    
    rst = self.dim.vector(a)
    return rst
  
  def maxUnpool2d(self,inputs,indices,ks):
    if (len(inputs.shape)!=4):
       raise Exception("input({})不符合[miniBatch*inChannels*H*W]的形状要求".format(inputs.shape)) 
    indices=self.dim.vector(indices)
    if (inputs.shape!=indices.shape):
      raise Exception("input({})与indices({})的形状不一致".format(inputs.shape,indices.shape))
    a=[]
    ks=int(ks)
    for i,channel in enumerate(inputs):
      a.append([])
      for j,kernel in enumerate(channel):
        a[i].append([])
        h=kernel.shape[0]
        w=kernel.shape[1]
        q=[]
        for k in range(h):
          p=[]
          for l in range(w):
            factor=self.dim.zeros([ks,ks])
            r=math.floor(indices[i,j,k,l]/ks)
            c=math.floor(indices[i,j,k,l]%ks)
            factor[r,c]=1
            p.append(kernel[k,l]*factor)
          q.append(self.dim.concat(p,1))
        a[i][j]=self.dim.concat(q,0)
    rst = self.dim.vector(a)
    return rst
  
  def avgUnpool2d(self,inputs,ks):
    if (len(inputs.shape)!=4):
       raise Exception("input({})不符合[miniBatch*inChannels*H*W]的形状要求".format(inputs.shape)) 

    ks=int(ks)
    a=[]
    factor = self.dim.fill(1/ks,(ks,ks))
    for i,channel in enumerate(inputs):
      a.append([])
      for j,kernel in enumerate(channel):
        a[i].append(self.dim.kron(kernel,factor))
    
    rst = self.dim.vector(a)
    return rst

  def maxUnpool3d(self,inputs,indices,ks):pass
  def avgUnpool3d(self,inputs,indices,ks):pass   

class NN(object):
  def __init__(self,dim):
    self.dim = dim
    self.grad = autograd
    self.Vector = dim.Vector
    #self.random=new Random(dim)
    self.functional = Functional(dim)
    self.Module = Module

  def Module1(self,moduleClass): return moduleClass()
  def Sequential(self,*modules): return Sequential(modules)
  def Linear(self,inF,outF,bias=True):return Linear(inF,outF,bias)
  def ReLU(self) : return ReLU()
  def Conv2d(self,inChannels,outChannels,kernelSize,stride=1,padding=0,bias=False):
    return Conv2d(inChannels,outChannels,kernelSize,stride,padding,bias)
  def MaxPool2d(self,ks,padding=0): return MaxPool2d(ks,padding=0)
  def CrossEntropyLoss(self): return CrossEntropyLoss()
  def MSELoss(self): return MSELoss()  

class Module:
  def __init__(self):
    self.moduleList=[]
    self.eps=1e-5
  def addModule(self,name,module):
    if (module==None):
      module=name    
      name = str(self.count)
    self.moduleList.append({"name":name,"module":module})
    self.count=len(self.moduleList)
  
  def flatten(self,arr):
    for a in arr:
      if isinstance(a, list):
        yield from self.flatten(a)
      else:
        yield a
  '''def print(self):
    string="Sequential(\n"
    string=string+self.moduleList.map(x=>`(${x.name}): ${x.module.print()}`).join('\n')
    string=string+"\n)"
    print(string)
  '''
  def _modules(self):
    for x in self.moduleList:
      if (len(x["module"].moduleList)!=0):
        yield from x["module"].modules()
      else:  
        yield x["module"] 
  def modules(self):
     mods=self.flatten(self._modules())
     return list(filter(lambda x:x!=None,mods))
  def _parameters(self):
    for x in self.moduleList:
      if (len(x["module"].moduleList)!=0):
        yield from x["module"].parameters()
      else:  
        yield getattr(x["module"],"params",None)  
  def parameters(self):
     params=self.flatten(self._parameters())
     return list(filter(lambda x:x!=None,params))
  def forward(self):
    print("must implement this function")
  
class Sequential(Module):
  def __init__(self,modules):
    super(Sequential,self).__init__()
    for i,x in enumerate(modules):
      self.moduleList.append({"name":str(i),"module":x})
    self.count=len(self.moduleList)
  
  def forward(self,x):
    for a in self.moduleList:
      x=a.module.forward(x)
    return x
  
  '''print(){
    let str
    str=`Sequential(\n`
    str=str+self.moduleList.map(x=>`(${x.name}): ${x.module.print()}`).join('\n')
    str=str+`\n)`
    console.log(str)
  }
  '''

class Linear(Module):
  def __init__(self,inF,outF,bias=True):
    super(Linear,self).__init__()
    self.ins = inF
    self.out = outF
    self.bias = bias
    self.params=[]
  def forward(self,input):
    if (input.shape[1]!=self.ins or input.ndim!=2): raise Exception("参数[{}]不符合要求{},{}".format(input.shape,self.ins,input.ndim))
    self.input=input
    if (self.weight==None):
      self.weight=dim.vector(dim.random.random((self.input.shape[1],self.out)))
      self.weight.setGrad()
      self.params.append(self.weight)
    
    if (self.B==None and self.bias):
      self.B=dim.vector(dim.random.random((self.input.shape[0],self.out)))
      self.B.setGrad()
      self.params.append(self.B)
    
    if (self.bias): return self.input.dot(self.weight).add(self.B)
    return self.input.dot(self.weight)    
  
  def print(self):
    return "Linear(in_features={}, out_features={}, bias={})".format(self.ins,self.out,self,bias)

class ReLU(Module):
  def __init__(self):
    super(ReLU,self).__init__()
  def forward(self,x):
    return dim.nn.functional.relu(x)
  def print(self):
    return "ReLu()"

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
  
  def forward(self,inputs):
    self.Input=inputs
    if (self.Filter==None):
      self.Filter=dim.vector(dim.random.random((self.outChannels,self.inChannels,
                                    self.kernelSize,self.kernelSize)))
      self.Filter.setGrad()
      self.params.append(self.Filter)
    
    if (self.B==None and self.bias):
      self.B=dim.vector(dim.random.random((inputs.shape[0],self.outChannels,self.kernelSize)))
      self.B.setGrad()
      self.params.append(self.B)
  
    if (self.bias): return dim.nn.functional.conv2d(self.Input,self.Filter,self.stride,self.padding).add(self.B)
    return dim.nn.functional.conv2d(self.Input,self.Filter,self.stride,self.padding)
  
  def print(self):
    return "Conv2d(inChannels={}, outChannels={}, kernelSize={},stride={},padding={},bias={})".format(self.inChannels,self.outChannels,self.kernelSize,self.stride,self.padding,self.bias)

class MaxPool2d(Module):
  def __init__(self,ks,padding):
    super(MaxPool2d,self).__init__()
    self.ks= ks
    self.indices = []
    self.padding = padding
    self.params=[]
  
  def forward(self,x):
    self.X=x
    self.result = dim.nn.functional.maxPool2d(x,self.ks,self.indices,self.padding)
    return self.result
  
  def print(self):
    return "MaxPool2d(kernelSize={}, padding={})".format(self.ks,self.padding)

class CrossEntropyLoss(Module):
  def __init__(self):
    super(CrossEntropyLoss,self).__init__()
  def forward(self,x,y):
    return dim.nn.functional.crossEntropy(x,y)
  def print(self):
    return "CrossEntropyLoss()"

class MSELoss(Module):
  def __init__(self):
    super(MSELoss,self).__init__()
  def forward(self,x,y):
    return dim.nn.functional.mseLoss(x,y)
  def print(self):
    return "MSELoss()"


class Optimizer:
  def __init__(self,params=None):
    self.Optimizer(params)

  def Optimizer(self,params):
    if (params and (not isinstance(params,list))): params=[params]
    if (params):
      self.params = params
    return self
  def step(self):
    for x in self.params:
      if (x.requiresGrad):
        x.sub_(x.grad.mul(self.lr))
  def zeroGrad(self):
    for x in self.params:
      if x.requiresGrad:
        x.gradClear()
  def Adam(self,params,args):
    if (params and (not isinstance(params,list))): params=[params]
    if (not params): params=self.params
    return Adam(params,args)

class Adam(Optimizer):
  def __init__(self,params,args={}):
    super(Adam,self).__init__()
    if (params):
      self.params = params
    self.lr  = args.lr or 0.001
    self.rho = args.rho or 0.9
    self.eps = args.eps or 1e-08
    self.weight_decay = args.weight_decay or 0
