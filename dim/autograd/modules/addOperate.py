import dim
from ..operate import Operate
from ..constant import Constant

class AddOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(AddOperate,self).__init__(left,right,"add",args,name)

  def partGrad(self,partial,prevOp):
    if (partial.type not in ["Variable","Operate"]): raise Exception("partial参数必须是Variable、Operate类型")
    if (self.catch and self._grads.get(partial.name,None)): 
      return self._grads[partial.name]
    if (prevOp is None):  
      data=self.eval()
      if isinstance(data,dim.Vector):
        prevOp = Constant(dim.ones(data.shape))
      else:
        prevOp = Constant(dim.cl.ones(data.shape))
    part1=self.left.partGrad(partial,prevOp)
    part2=self.right.partGrad(partial,prevOp)
  
    rst=dim.autograd.AddOperate.wrapper(part1,part2)
    self._grads[partial.name]=rst
    
    return rst

  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "("+self.left.expression() + "+" + self.right.expression()+")"
    self._expressionStr = rst
    return rst
  
  def eval(self,useCatch=True):
    if (useCatch and self.catch and self._data is not None): return self._data
    
    rst = self.left.eval(useCatch)+self.right.eval(useCatch)
    self._data = rst
    return rst

  @staticmethod
  def wrapper(left,right,args=None,name=None):
    if (left.type=="Constant" and left.number==0): return right
    if (right.type=="Constant" and right.number==0): return left
    '''if (left.type=="Constant" and (left.data==0).all()): return right
    if (right.type=="Constant" and (right.data==0).all()): return left
    if (left.type=="Constant" and right.type=="Constant"): return Constant(dim.add(left.data,right.data))
    '''
    return AddOperate(left,right,args,name)
