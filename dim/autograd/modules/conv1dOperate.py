import dim
from ..operate import Operate
from ..constant import Constant

class Conv1dOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(Conv1dOperate,self).__init__(left,right,"conv1d",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))
    
    dLeft=dim.autograd.ConvTranspose1dOperate.wrapper(prevOp,self.right,self.args)
    temp1 = prevOp.eval().swapaxes(0,1)
    temp2 = self.left.eval().swapaxes(0,1)
    temp3 = dim.nn.functional.conv1d(temp2,temp1)
    dRight = Constant(temp3.swapaxes(0,1))
    if (self.left.name==partial.name):
      part1 = dLeft
      part2=self.right.partGrad(partial,dRight)
    elif (self.right.name==partial.name):
      part1 = dRight
      part2=self.left.partGrad(partial,dLeft)
    else:
      part1=self.left.partGrad(partial,dLeft)
      part2=self.right.partGrad(partial,dRight)
  
    part3=dim.autograd.AddOperate.wrapper(part1,part2)
    rst = part3
    self._grads[partial.name]=rst
    return rst  
  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "conv1d("+self.left.expression()+","+self.right.expression()+")"
    self._expressionStr = rst
    return rst
  
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst=dim.nn.functional.conv1d(self.left.eval(),self.right.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return Conv1dOperate(left,right,args,name)
