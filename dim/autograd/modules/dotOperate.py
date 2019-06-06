#coding utf-8
import dim
from ..operate import Operate
from ..constant import Constant

class DotOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(DotOperate,self).__init__(left,right,"dot",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)):
      #print("use catch",partial.name,self._grads[partial.name])
      return self._grads[partial.name]
    if (prevOp is None):  
      data=self.eval()
      if isinstance(data,dim.Vector):
        prevOp = Constant(dim.ones(data.shape))
      else:
        prevOp = Constant(dim.cl.ones(data.shape))
    '''直接求eval的方式也没有问题，因为上一层的反算结果已经完成了。这种方式的问题是
       不能在总算式的gradFn.gradExpression()中看到各偏导的dot计算公式
    '''
    #dLeft = Constant(prevOp.eval().dot(self.right.eval().T))
    #dRight = Constant(self.left.eval().T.dot(prevOp.eval()))
        
    if (self.catch and self._grads.get(self.left.name,None)): 
      #print("catch left")
      dLeft=self._grads.get(self.left.name)
    else:
      #tRight = dim.autograd.TOperate.wrapper(self.right,None)
      tRight = Constant(self.right.eval().T)
      dLeft = dim.autograd.DotOperate.wrapper(prevOp,tRight)
      #print("catch left",self.left.name)
      #dLeft = Constant(prevOp.eval().dot(self.right.eval().T))
      self._grads[self.left.name]=dLeft

    if (self.catch and self._grads.get(self.right.name,None)): 
      #print("catch right")
      dRight=self._grads.get(self.right.name)
    else:
      #print("catch right",self.right.name)
      #tLeft = dim.autograd.TOperate.wrapper(self.left,None)
      tLeft = Constant(self.left.eval().T)
      dRight = dim.autograd.DotOperate.wrapper(tLeft,prevOp)
      #dRight = Constant(self.left.eval().T.dot(prevOp.eval()))
      self._grads[self.right.name]=dRight
    
    if (self.left.name==partial.name):
      part1 = dLeft
      part2 = self.right.partGrad(partial,dRight)
    elif (self.right.name==partial.name):
      part1 = self.left.partGrad(partial,dLeft)
      part2 = dRight
    else:
      part1=self.left.partGrad(partial,dLeft)
      part2=self.right.partGrad(partial,dRight)
    
    rst=dim.autograd.AddOperate.wrapper(part1,part2)
    self._grads[partial.name]=rst
    return rst  

  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    part1=self.left.expression()
    part2=self.right.expression()
    rst = "({}@{})".format(part1,part2)
    self._expressionStr = rst
    return rst
 
  def eval(self,useCatch=True):
    if (useCatch and self.catch and self._data is not None): 
      return self._data
    data=self.left.eval(useCatch)
    if isinstance(data,dim.Vector):
      rst = data.dot(self.right.eval(useCatch))
    else:
      rst = data.mm(self.right.eval(useCatch))
    self._data = rst
    return rst

  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return DotOperate(left,right,args,name)
