# OpenCL Matrix Multiplication Test
# By: J. Hunter Moore
# Date: 21 November 2018
#
# This program serves an exploration into OpenCL via the pyopencl module.
# This program generated two random matrices, then multiplies them, and
# shows the time taken to do so. To compare results, a serial matrix
# multiplcation method has been commented out at the bottom of this program
#
# Credits to: PyOpenCL documentation (https://documen.tician.de/pyopencl)
#             inducer on github (https://github.com/inducer/pyopencl)
#             scipy's numpy reference (https://docs.scipy.org/doc/numpy/reference/)
#             AndreasKloeckner on wiki.tiker.net (https://wiki.tiker.net/PyOpenCL)

import pyopencl.array as cl_array
import pyopencl.tools as cl_tools
import pyopencl as cl
import numpy as np
import time

#Change as desired.  Ensure that you have enough memory to suport a matrix
# NxN, else you will fill memory and crash
MATRIX_SIZE = 1000

#This defines the global work group size  Your device may either use a 64-bit
#or 32-bit.  If result values are strange, try switching to 32-bit
DIMENSIONS = np.array([MATRIX_SIZE,MATRIX_SIZE],dtype=np.int64)
#DIMENSIONS = np.array([MATRIX_SIZE,MATRIX_SIZE],dtype=np.int32)

np.random.seed(9353)

# Create the source matrices - make sure dtype matches what is in the kernel
m1 = np.array(np.random.randint(10, size=DIMENSIONS),dtype=np.int32)
m2 = np.array(np.random.randint(10, size=DIMENSIONS),dtype=np.int32)
m1=np.random.randn(1000,1000)
m2=np.random.randn(1000,1000)
# This will only be used to create the cl-type matrix. This will set dtype
result = np.zeros_like(m1)

print("Matrices Initialized")

# Get platforms, both CPU and GPU
plat = cl.get_platforms()
CPU = plat[0].get_devices()
try:
    GPU = plat[1].get_devices()
except IndexError:
    GPU = "none"

#Create context for GPU/CPU (largely automated by pyopenCL)
if GPU!= "none":
    ctx = cl.Context(GPU)
else:
    ctx = cl.Context(CPU)

print(ctx)
# Create queue for each kernel execution
queue = cl.CommandQueue(ctx)

# get memory flags
mf = cl.mem_flags

print("Context and Queue initialized")

#Copy everything into the device
  #Since pyopencl has nice methods for array handling, we will use them on our matrices
m1_g = cl_array.to_device(queue, m1)
m2_g = cl_array.to_device(queue, m2)
print(m1_g.queue.device)
result_g = cl_array.to_device(queue, result)
  #for a single integer, we will need to load into the Buffer with the proper flags and data type
size_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(MATRIX_SIZE))

print("Matrices moved to GPU")

#This is the C code for the kernel, compiled at runtime by openCL
src = '''
__kernel void matrixMult(__global int *m1, __global int *m2, __global int *result, __global int *width)
{
    int w = *width;
    int posx = get_global_id(1);
    int posy = get_global_id(0);
    for(int i = 0; i < w; i++)
        result[posy*w+posx] += m1[posy*w+i]*m2[i*w+posx];
}'''

#build the code from src
prg = cl.Program(ctx, src).build()

print("Program created.  Getting ready to start multiplying.")

#Mark start time
start_time = time.time()
ev = prg.matrixMult(queue, DIMENSIONS, None, m1_g.data, m2_g.data, result_g.data, size_g)

#Hold main until queue has emptied
queue.finish()

#get total time
finish_time = time.time() - start_time

print("Finished Multiplying in: " + str(finish_time))

#If matrices are of reasonable size, display
print(m1)
print(m2)
print(result_g)        #show result_g result hasn't been passed the finished matrix

#====================================================
#   SerialMethod
#====================================================
#
#print("Starting serial method: ")

#start_time = time.time()

#for i in range(MATRIX_SIZE):
#    for j in range(MATRIX_SIZE):
#        for k in range(MATRIX_SIZE):
#            result[i][j] += m1[i][k] * m2[k][j]

#finish_time = time.time() - start_time
#print("Time to finish serial method: " + str(finish_time))

start_time = time.time()
r2=m1.dot(m2)
print(r2)
finish_time = time.time() - start_time
print("Time to finish serial method: " + str(finish_time))
