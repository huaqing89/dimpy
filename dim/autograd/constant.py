import json
import math
import random
import hashlib

import dim
from .autograd import Autograd

CONSTANT= []

class Constant(Autograd):
  def __init__(self,data):
    super(Constant,self).__init__()
    if isinstance(data,Constant): return self
    if self.isNumber(data): data=dim.vector(data)
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
    self.name = "const"+str(random.random())[-6:]
    self.data = data
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
