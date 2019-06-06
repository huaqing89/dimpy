__kernel void mm(__global float *a,
                 __global float *b,
                 __global float *o,
                 int w1,
                 int w2
                )
  { 
    int posx=get_global_id(1);
    int posy=get_global_id(0);
    for (int i=0;i<w1;i++)
      o[posy*w2+posx] += a[posy*w1+i]*b[i*w2+posx]; 
  }

__kernel void conv2d(__global float *a,
                 __global float *b,
                 __global float *o,
                 int w1,
                 int w2,
                 int hf,
                 int wf
                )
  { 
    int posx=get_global_id(1);
    int posy=get_global_id(0);
    for (int i=0;i<hf;i++)
      for (int j=0;j<wf;j++)
        //o[posy*w1+posx] += 0;
        o[posy*w1+posx] += a[(posy+i)*w2+posx+j]*b[i*wf+j]; 
  }
