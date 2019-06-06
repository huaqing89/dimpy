#coding:utf-8
import dim

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
    return isinstance(val,(int,float))
  def shrink(self): pass
  
  def factor(self,opStr): return [self]
  def gradExpression(self):
    m=[]
    for x in self._grads:
      m.append({"name":x,"expression":self._grads[x].expression()})
    return m
