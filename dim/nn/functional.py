#coding:utf-8
import math
import dim
from ..vector import Vector

def softmax(x,axis=1):
  x=dim.vector(x)
  y=x.exp()
  total=y.sum(axis,keepdims=True)
  rst=y/total
  if (axis==1):
    rst = x.setGradFn(rst,"softmax")    
  return rst

def softmaxDeri(x,a):
  x,a=dim.vector(x,a)
  argmax=a.argmax(1).value()
  data=x.value
  '''rst = Vector(data.map((y,i)=>{
    return y.map((z,j)=>argmax[i]==j?z*(1-z):-z*y[argmax[i]])
  }))
  '''
  return rst
#Activation Function
def relu(x):
  rst=x.copy()
  rst[x<0]=0
  rst = x.setGradFn(rst,"relu")
  return rst

def reluDeri(x):
  rst= x.copy()
  rst[x>0]=1
  rst[x<0]=0
  return rst

def relu6(x):
  rst=x.copy()
  rst[x>6]=6
  rst[x<0]=0
  rst = x.setGradFn(rst,"relu6")
  return rst

def relu6Deri(x):
  rst = x.copy()
  rst[x>6]=1
  rst[x<=6]=0
  return rst

def softplus(x):
  x=dim.vector(x)
  return (x.exp()+1).log()
  
def sigmoid(x):
  x=dim.vector(x)
  return 1/(1+(-x).exp())

def tanh(x):
  x=dim.vector(x)
  return x.tanh()
  
def dropout(a,keep):
  if (keep<=0 or keep>1): raise Exception("keep_prob参数必须属于(0,1]")
  a=dim.vector(a)
  arr=[]
  '''return new Vector(a.data.map((x,i)=>{
    if (x instanceof Vector) return dropout(x,keep)
    if (i==0){
      let remain=a.data.length*keep
      for (let j=0;j<a.data.length;j++) arr.append(j)
      arr = random.shuffle(arr).slice(0,remain)
    }
    return (arr.indexOf(i)>=0)?x/keep:0
  }))
  '''

#Loss Function
def mseLoss(a,y):
  #also named L2
  #a,y=dim.vector(a,y)
  #return y.sub(a).square().mean()
  d=y-a
  return (d*d).mean()
def binaryCrossEntropy(a,y):
  a,y=dim.vector(a,y)
  return (y*a.log() + (1-y)*(1-a).log()).sum()

def crossEntropy(a,y):
  a,y=dim.vector(a,y)
  b=softmax(a,1)
  y_onehot=y.onehot(b.shape[1])

  rst = y_onehot.mul(b.log()).sum(1).neg().mean()
  rst = a.setGradFn(rst,"crossEntropy",right=y)
  return rst

def crossEntropyDeri(a,y):
  a,y=dim.vector(a,y)
  b=softmax(a,1)
  y_onehot=dim.onehot(y,a.shape[1])
  rst = b.sub(y_onehot).div(b.shape[0])
  return rst

def logcoshLoss(a,y):
  a,y=dim.vector(a,y)
  return y.sub(a).cosh().log().sum()

#cnn function
def conv1d(inputs, filters, stride=1, padding=0):
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
  rst=dim.vector(a)
  #P1=(((In-1)*S+F) -(((In-F+2*P0)/S)+1))/2let In=input.shape[2]
  In=inputs.shape[2]
  F=filters.shape[2]
  S=stride
  P0=padding
  gradPadding=math.floor(((((In-1)*S+F) -(((In-F+2*P0)/S)+1))/2))
  rst = inputs.setGradFn(rst,"conv1d",left=inputs,right=filters,args={"padding":gradPadding})
  return rst

def conv2d(inputs, filters, stride=1, padding=0):
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
  rst = dim.vector(a)
  #P1=(((In-1)*S+F) -(((In-F+2*P0)/S)+1))/2let In=input.shape[2]
  In=inputs.shape[2]
  F=filters.shape[2]
  S=stride
  P0=padding
  gradPadding=int((((In-1)*S+F) -(((In-F+2*P0)/S)+1))/2)
  
  rst = inputs.setGradFn(rst,"conv2d",left=inputs,right=filters,args={"padding":gradPadding})
  return rst
  
def conv3d(inputs, filters, stride=1, padding=0):
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
        d=math.floor((ideep-fdeep)/stride+1)
        h=math.floor((ih-fh)/stride+1)
        w=math.floor((iw-fw)/stride+1)
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
  return dim.vector(a)
       
def convTranspose1d(inputs, filters, stride=1, padding=0):
  if (len(inputs.shape)!=3):
    raise Exception("input({})不符合[miniBatch*inChannels*W]的形状要求".format(inputs.shape)) 
  if (len(filters.shape)!=3):
    raise Exception("filter({})不符合[inChannels*outChannels*W]的形状要求".format(filters.shape)) 
  if (inputs.shape[1]!=filters.shape[0]): 
    raise Exception("input({})与filter({})中channels数不一致".format(inputs.shape,filters.shape))
  #change channel
  filters = dim.swapaxes(filters,0,1)
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
  rst = dim.vector(a)
  return rst
  
def convTranspose2d(inputs, filters, stride=1, padding=0):
  #要实现还原运算，padding=((Out-1)*stride-Input+Filter)/2
  if (len(inputs.shape)!=4):
    raise Exception("input({})不符合[miniBatch*inChannels*H*W]的形状要求".format(inputs.shape)) 
  if (len(filters.shape)!=4):
    raise Exception("filter({})不符合[outChannels*inChannels*H*W]的形状要求".format(filters.shape)) 
  if (inputs.shape[1]!=filters.shape[0]):
    raise Exception("input({})与filter({})中channels数不一致".format(inputs.shape,filters.shape))
  filters = dim.swapaxes(filters,0,1)
  a=[]
  for i in range(inputs.shape[0]):
    a.append([])
    for j in range(filters.shape[0]):
       a[i].append([])
       for k in range(filters.shape[1]):
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
  return dim.vector(a)
  
def convTranspose3d(inputs, filters, stride=1, padding=0):
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
    for j in range(filters.shape[0]):
       a[i].append([])
       for k in range(filters.shape[1]):
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
  return dim.vector(a)
  
#Pool Function
def maxPool1d(inputs,ks,padding=0,includeIndices=False):
  if (len(inputs.shape)!=3):
     raise Exception("input({})不符合[miniBatch*inChannels*W]的形状要求".format(inputs.shape)) 
  
  ks=int(ks)
  a=[]
  indices=[]
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
  rst = dim.vector(a)
  rst = inputs.setGradFn(rst,"maxPool1d",left=inputs,right=ks,args={"indices":indices})
  if (includeIndices): return rst,indices
  return rst
  
def avgPool1d(inputs,ks,padding=0):
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
  rst = dim.vector(a)
  rst = inputs.setGradFn(rst,"avgPool1d",left=inputs,right=ks)
  return rst
def maxPool2d(inputs,ks,padding=0,includeIndices=False):
  if (len(inputs.shape)!=4):
     raise Exception("input({})不符合[miniBatch*inChannels*H*W]的形状要求".format(inputs.shape)) 
  
  ks=int(ks)
  a=[]
  indices=[]
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
  
  rst =  dim.vector(a)
  rst =  inputs.setGradFn(rst,"maxPool2d",left=inputs,right=ks,args={"indices":indices})
  if (includeIndices): return rst,indices
  return rst
  
def avgPool2d(inputs,ks,padding=0):
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
    
  rst =  dim.vector(a)
  rst =  inputs.setGradFn(rst,"avgPool2d",left=inputs,right=ks)
  return rst

def maxPool3d(self):pass
def avgPool3d(self):pass

def maxUnpool1d(inputs,indices,ks):
  if (len(inputs.shape)!=3):
     raise Exception("input({})不符合[miniBatch*inChannels*W]的形状要求".format(inputs.shape)) 
  print("maxUnpool1d indices:",indices)
  indices=dim.vector(indices)
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
        factor = dim.zeros(ks)
        r=math.floor(indices[i,j,k]%ks)
        factor[r]=1
        p.append(kernel[k]*factor)
      a[i][j]=dim.concat(p,0)
  
  rst = dim.vector(a)
  return rst
def avgUnpool1d(inputs,ks):
  if (len(inputs.shape)!=3):
     raise Exception("input({})不符合[miniBatch*inChannels*W]的形状要求".format(inputs.shape)) 
  ks=int(ks)
  a=[]
  factor = dim.fill(1/ks,ks)
  for i,channel in enumerate(inputs):
    a.append([])
    for j,kernel in enumerate(channel):
      a[i].append(dim.kron(kernel,factor))
  
  rst = dim.vector(a)
  return rst

def maxUnpool2d(inputs,indices,ks):
  if (len(inputs.shape)!=4):
     raise Exception("input({})不符合[miniBatch*inChannels*H*W]的形状要求".format(inputs.shape)) 
  indices=dim.vector(indices)
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
          factor=dim.zeros([ks,ks])
          r=math.floor(indices[i,j,k,l]/ks)
          c=math.floor(indices[i,j,k,l]%ks)
          factor[r,c]=1
          p.append(kernel[k,l]*factor)
        q.append(dim.concat(p,1))
      a[i][j]=dim.concat(q,0)
  rst = dim.vector(a)
  return rst
  
def avgUnpool2d(inputs,ks):
  if (len(inputs.shape)!=4):
     raise Exception("input({})不符合[miniBatch*inChannels*H*W]的形状要求".format(inputs.shape)) 

  ks=int(ks)
  a=[]
  factor = dim.fill(1/ks,(ks,ks))
  for i,channel in enumerate(inputs):
    a.append([])
    for j,kernel in enumerate(channel):
      a[i].append(dim.kron(kernel,factor))
  
  rst = dim.vector(a)
  return rst

def maxUnpool3d(inputs,indices,ks):pass
def avgUnpool3d(inputs,indices,ks):pass   
