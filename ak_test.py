#!/usr/bin/env python3
# coding: utf-8

# In[1]:


import importlib
import numpy as np
import math
import gc
import sys

# In[2]:


import arkouda as ak


# In[3]:


ak.v = False
if len(sys.argv) > 1:
    ak.connect(server=sys.argv[1], port=sys.argv[2])
else:
    ak.connect()


# In[5]:


a = ak.randint(0,2,10,dtype=ak.bool)
a[1] = True
a[2] = True
print(a)


# In[6]:


a = ak.ones(10,dtype=ak.int64)
iv = ak.arange(0,10,1)[::2]
a[iv] = 10
print(a)


# In[7]:


a = ak.ones(10)
iv = ak.arange(0,10,1)[::2]
a[iv] = 10.0
print(a)


# In[8]:


a = ak.ones(10,dtype=ak.bool)
iv = ak.arange(0,10,1)[::2]
a[iv] = False
print(a)


# In[9]:


a = ak.ones(10,dtype=ak.bool)
iv = ak.arange(0,5,1)
b = ak.zeros(iv.size,dtype=ak.bool)
a[iv] = b
print(a)


# In[10]:


a = ak.randint(10,20,10)
print(a)
iv = ak.randint(0,10,5)
print(iv)
b = ak.zeros(iv.size,dtype=ak.int64)
a[iv] = b
print(a)


# In[11]:


#ak.disconnect()


# In[12]:


ak.v = False
a = ak.randint(10,30,40)
vc = ak.value_counts(a)
print(vc[0].size,vc[0])
print(vc[1].size,vc[1])


# In[13]:


ak.v = False

a = ak.arange(0,10,1)
b = a[a<5]
a = ak.linspace(0,9,10)
b = a[a<5]
print(b)


# In[14]:


ak.v = True
ak.pdarray_iter_thresh = 1000
a = ak.arange(0,10,1)
print(list(a))


# In[15]:


ak.v = False
a = ak.randint(10,30,40)
u = ak.unique(a)
h = ak.histogram(a,bins=20)
print(a)
print(h.size,h)
print(u.size,u)


# In[16]:


ak.v = False
a = ak.randint(10,30,50)
h = ak.histogram(a,bins=20)
print(a)
print(h)


# In[17]:


ak.v = False
a = ak.randint(0,2,50,dtype=ak.bool)
print(a)
print(a.sum())


# In[18]:


ak.v = False
a = ak.linspace(101,102,100)
h = ak.histogram(a,bins=50)
print(h)


# In[19]:


ak.v = False
a = ak.arange(0,100,1)
h = ak.histogram(a,bins=10)
print(h)


# In[20]:


ak.v = False
a = ak.arange(0,99,1)
b = a[::10] # take every tenth one
b = b[::-1] # reverse b
print(a)
print(b)
c = ak.in1d(a,b) # put out truth vector
print(c)
print(a[c]) # compress out false values


# In[21]:


ak.v = False
a = np.ones(10,dtype=np.bool)
b = np.arange(0,10,1)

np.sum(a),np.cumsum(a),np.sum(b<4),b[b<4],b<5


# In[22]:


ak.v = False
# currently... ak pdarray to np array
a = ak.linspace(0,9,10)
b = np.array(list(a))
print(a,a.dtype,b,b.dtype)

a = ak.arange(0,10,1)
b = np.array(list(a))
print(a,a.dtype,b,b.dtype)

a = ak.ones(10,dtype=ak.bool)
b = np.array(list(a))
print(a,a.dtype,b,b.dtype)


# In[23]:


ak.v = False

b = np.linspace(1,10,10)
a = np.arange(1,11,1)
print(b/a)


# In[24]:


ak.v = False

#a = np.ones(10000,dtype=np.int64)
a = np.linspace(0,99,100)
#a = np.arange(0,100,1)
print(a)


# In[25]:


ak.v = False
print(a.__repr__())
print(a.__str__())


# In[26]:


ak.v = False
print(a)
print(type(a), a.dtype, a.size, a.ndim, a.shape, a.itemsize)


# In[27]:


ak.v = False
a = ak.arange(0,100,1)


# In[28]:


ak.v = False
print(a)


# In[29]:


ak.v = False

b = ak.linspace(0,99,100)
print(b.__repr__())


# In[30]:


ak.v = False
b = ak.linspace(0,9,10)
a = ak.arange(0,10,1)
print(a.name, a.size, a.dtype, a)
print(b.name, b.size, b.dtype, b)
print(ak.info(a+b))


# In[31]:


ak.v = False

c = ak.arange(0,10,1)
print(ak.info(c))
print(c.name, c.dtype, c.size, c.ndim, c.shape, c.itemsize)
print(c)


# In[32]:


ak.v = False

print(5+c + 5)


# In[33]:


c = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
c = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19.1])
print(c.__repr__(), c.dtype.__str__(), c.dtype.__repr__())


# In[34]:


ak.v = False
a = np.ones(9)
b = np.arange(1,10,1)
print(a.dtype,b.dtype)
c = ak.ones(9)
d = ak.arange(1,10,1)
print(c.dtype,d.dtype)
y = a/b
z = c/d
print("truediv  \nnp out:",y,"\nak out:",z)
print(y[5],z[5],y[5] ==z[5])
y = a//b
z = c//d
print("floordiv \nnp out:",y,"\nak out:",z)
print(y[5],z[5],y[5] ==z[5])


# In[35]:


ak.v = False

c = ak.arange(1,10,1)
c //= c
print(c)
c += c
print(c)
c *= c
print(c)


# In[36]:


ak.v = False

a = np.ones(9,dtype=np.int64)
b = np.ones_like(a)
print(b)


# In[37]:


ak.v = False

a = ak.ones(9,dtype=ak.int64)
b = ak.ones_like(a)
print(b)


# In[38]:


ak.v = False

a = ak.arange(0,10,1)
b = np.arange(0,10,1)

print(a[5] == b[5])


# In[39]:


ak.v = False

a = ak.arange(0,10,1)
b = np.arange(0,10,1)

a[5] = 10.2
print(a[5])


# In[40]:


ak.v = False

a = ak.arange(0,10,1)
b = np.arange(0,10,1)
#print((a[:]),b[:])
#print(a[1:-1:2],b[1:-1:2])
#print(a[0:10:2],b[0:10:2])
print(a[4:20:-1],b[4:20:-1])
print(a[:1:-1],b[:1:-1])


# In[ ]:





# In[41]:


ak.v = False
d = ak.arange(1,10,1)
#d.type.__class__,d.name,d.isnative,np.int64.__class__,bool
ak.info(d)
#dir(d)


# In[42]:


ak.v = False

a = ak.ones(10,dtype=ak.bool)
print(a[1])


# In[43]:


ak.v = False

a = ak.zeros(10,dtype=ak.bool)
print(a[1])


# In[44]:


ak.v = False
a = ak.ones(10,dtype=ak.bool)
a[4] = False
a[1] = False
print(a)
print(a[::2])
print(a[1])
a = ak.ones(10,dtype=ak.int64)
a[4] = False
a[1] = False
print(a)
print(a[::2])
print(a[1])
a = ak.ones(10)
a[4] = False
a[1] = False
print(a)
print(a[::2])
print(a[1])


# In[45]:


ak.v = False

a = ak.arange(0,10,1)
b = list(a)
print(b)

a = a<5
b = list(a)
print(b)


# In[46]:


ak.v = False
a = ak.linspace(1,10,10)
print(ak.abs(a))
print(ak.log(a))
print(ak.exp(a))
a.fill(math.e)
print(ak.log(a))


# In[47]:


type(bool),type(np.bool),type(ak.bool),type(True)


# In[48]:


ak.v = False
a = ak.linspace(0,9,10)
print(a,ak.any(a),ak.all(a),ak.all(ak.ones(10,dtype=ak.float64)))
b = a<5
print(b,ak.any(b),ak.all(b),ak.all(ak.ones(10,dtype=ak.bool)))
c = ak.arange(0,10,1)
print(c,ak.any(c),ak.all(c),ak.all(ak.ones(10,dtype=ak.int64)))
print(a.any(),a.all(),b.any(),b.all())


# In[49]:


ak.v = False
a = ak.linspace(0,9,10)
ak.sum(a)
b = np.linspace(0,9,10)
print(ak.sum(a) == np.sum(b),ak.sum(a),np.sum(b),a.sum(),b.sum())


# In[50]:


ak.v = False
a = ak.linspace(1,10,10)
b = np.linspace(1,10,10)
print(ak.prod(a) == np.prod(b),ak.prod(a),np.prod(b),a.prod(),b.prod())


# In[51]:


ak.v = False

a = np.arange(0,20,1)
b = a<10
print(b,np.sum(b),b.sum(),np.prod(b),b.prod(),np.cumsum(b),np.cumprod(b))
print()
b = a<5
print(b,np.sum(b),b.sum(),np.prod(b),b.prod(),np.cumsum(b),np.cumprod(b))
print()
a = ak.arange(0,20,1)
b = a<10
print(b,ak.sum(b),b.sum(),ak.prod(b),b.prod(),ak.cumsum(b),ak.cumprod(b))
b = a<5
print(b,ak.sum(b),b.sum(),ak.prod(b),b.prod(),ak.cumsum(b),ak.cumprod(b))


# In[52]:


ak.v = False
a = ak.arange(0,10,1)
iv = a[::-1]
print(a,iv,a[iv])


# In[53]:


ak.v = False
a = ak.arange(0,10,1)
iv = a[::-1]
print(a,iv,a[iv])


# In[54]:


ak.v = False
a = ak.linspace(0,9,10)
iv = ak.arange(0,10,1)
iv = iv[::-1]
print(a,iv,a[iv])


# In[55]:


ak.v = False
a = np.arange(0,10,1)
iv = a[::-1]
print(a,iv,a[iv])


# In[56]:


ak.v = False
a = ak.arange(0,10,1)
b = a<20
print(a,b,a[b])


# In[57]:


ak.v = False
a = ak.arange(0,10,1)
b = a<5
print(a,b,a[b])


# In[58]:


ak.v = False
a = ak.arange(0,10,1)
b = a<0
print(a,b,a[b])


# In[59]:


ak.v = False
a = ak.linspace(0,9,10)
b = a<5
print(a,b,a[b])


# In[60]:


ak.v = False
N = 2**23 # 2**23 * 8 == 64MiB
A = ak.linspace(0,N-1,N)
B = ak.linspace(0,N-1,N)

C = A+B
print(ak.info(C),C)


# In[61]:


# turn off verbose messages from arkouda package
ak.v = False
# set pdarray_iter_thresh to 0 to only print the first 3 and last 3 of pdarray
ak.pdarray_iter_thresh = 0
a = ak.linspace(0,9,10)
b = a<5
print(a)
print(b)
print(a[b])
print(a)


# In[62]:



a = np.linspace(0,9,10)
b = a<5
print(a)
print(b)
print(a[b])
print(a)


# In[63]:


ak.v = False
ak.pdarray_iter_thresh = 0

a = ak.ones(10,ak.int64)
b = a | 0xff
print(a, b, a^b, b>>a, b<<1|1, 0xf & b, 0xaa ^ b, b ^ 0xff)
print(-a,~(~a))


# In[64]:


a = ak.ones(10,dtype=ak.int64)
b = ~ak.zeros(10,dtype=ak.int64)
print(~a, -b)


# In[65]:


a = np.ones(10,dtype=np.int64)
b = ~np.zeros(10,dtype=np.int64)
print(~a, -b)


# In[ ]:





# In[ ]:





# In[66]:


ak.shutdown()


# In[ ]:




