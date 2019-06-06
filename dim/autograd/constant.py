import json
import math

import dim
from .autograd import Autograd

class Constant(Autograd):
  sequence=0
  def __init__(self,data,name=None):
    super(Constant,self).__init__()
    if self.isNumber(data):
      self.number=data 
      self.data = dim.vector(data)
    elif (isinstance(data,dim.Vector) and data.size==1):
      self.number=data
      self.data = data
    elif (isinstance(data,dim.cl.Array) and data.size==1):
      self.number=data
      self.data = data
    else:
      self.number=None
      self.data = data
    '''节省空间，但严重影响效率
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
    '''
    if (name is None):
      Constant.sequence+=1
      self.name = "const"+str(Constant.sequence)
    else:
      self.name = name
    #print(self.name,self.number,data,type(data))
    self.type = "Constant"
    self._expressionStr=self.name
  
  def partGrad(self,partial=None,prevOp=None):
    rst = Constant(0)
    self._grads[self.name]=rst
    return rst

  def expression(self):
    if self.isNumber(self.data): return str(self.data)
    rst=json.dumps(self.data.tolist())
    if (len(rst)>50): 
      return "{}[{}]".format(self.name,"*".join(str(i) for i in self.data.shape)) 
    else:
      return "{}({})".format(self.name,rst)

  def eval(self,useCatch=True):return self.data
  def backward(self):return self.partGrad()
  def variables(self,prevOp,partial):return []
  def isSame(self,a):
    if (not isinstance(a,Constant)): return False
    if (self.name==a.name and self.data==a.data): return True
    return Fasle
