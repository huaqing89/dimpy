import sys
sys.path.append('..')

import dim

a1=dim.arange(2*6).reshape(1,2,6)
a1.setGrad()
print("a1",a1)
b1=dim.arange(2*2*3).reshape(2,2,3)
b1.setGrad()
print("b1",b1)

c1=dim.nn.functional.conv1d(a1,b1)
print("c=conv1d(a,b)",c1)
d1=dim.nn.functional.relu(c1)
e1=dim.nn.functional.maxPool1d(d1,2)
print("e=maxPool1d(d)",e1)
e1.backward()

print("a.grad",a1.grad)
print("b.grad",b1.grad)

a2=dim.arange(2*2*6*6).reshape(2,2,6,6)
a2.setGrad()
print("a",a2)
b2=dim.arange(3*2*3*3).reshape(3,2,3,3)
b2.setGrad()
print("b",b2)

c2=dim.nn.functional.conv2d(a2,b2)
print("c=conv1d(a,b)",c2)
d2=dim.nn.functional.relu(c2)
e2=dim.nn.functional.maxPool2d(d2,2)
print("e=maxPool1d(d)",e2)
e2.backward()

print("a.grad",a2.grad)
print("b.grad",b2.grad)

a3=dim.arange(2*2*3*6*6).reshape(2,2,3,6,6)
a3.setGrad()
print("a",a3)
b3=dim.arange(3*2*3*3*3).reshape(3,2,3,3,3)
b3.setGrad()
print("b",b3)

c3=dim.nn.functional.conv3d(a3,b3)
print("c=conv1d(a,b)",c3)
d3=dim.nn.functional.relu(c3)
e3=dim.nn.functional.maxPool3d(d3,3)
print("e=maxPool1d(d)",e3)
e3.backward()

print("a.grad",a3.grad)
print("b.grad",b3.grad)
