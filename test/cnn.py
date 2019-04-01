import sys
sys.path.append('..')

import autograd
import dim
dim=dim.Dim()
a=dim.arange(5).reshape(1,1,5)
a.setGrad()
print("a",a)
b=dim.arange(2).reshape(1,1,2)
b.setGrad()
print("b",b)

c=dim.nn.functional.conv1d(a,b)
print("c=conv1d(a,b)",c)
i=[]
d=dim.nn.functional.maxPool1d(c,2,i)
print("i:",i)
print("d=maxPool1d(c)",d)
d.backward()

print("a.grad",a.grad)
print("b.grad",b.grad)

x=dim.arange(2*5*5).reshape(1,2,5,5)
x.setGrad()
print("x",x)
y=dim.arange(2*3*3).reshape(1,2,3,3)
y.setGrad()
print("y",y)

z=dim.nn.functional.conv2d(x,y)
print("z=conv2d(x,y)",z)
i=[]
o=dim.nn.functional.maxPool2d(z,2,i)
print("i:",i)
print("o=maxPool2d(c)",o)
o.backward()

print("x.grad",x.grad)
print("y.grad",y.grad)
