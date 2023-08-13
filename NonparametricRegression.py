#!/usr/bin/env python
# coding: utf-8

# In[37]:


import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


# In[38]:


data_set = np.genfromtxt("hw04_data_set.csv", delimiter = "," , skip_header=1)
X = data_set[:,0]
Y = data_set[:,1]


# In[39]:


x_train =X[0:150]
x_test=X[150:272]
y_train =Y[0:150]
y_test= Y[150:272]
h=0.37


# In[40]:


def K(u):    #Kernel smoother
    return (1/math.sqrt((2*math.pi)))*np.exp(- (u**2)/2)
def W(u):   #Running mean smoother
    if np.abs(u) <= 0.5:
        return 1
    else:
        return 0
def B (x,x_i):
    h = 0.37
    left_borders = np.arange(1.5, 5.2, h)
    right_borders = np.arange(1.5 + h, 5.2 + h, h)


    for i in range(0,10):
        if  (x>left_borders[0] and x <= right_borders[0])   and (x_i>left_borders[0] and x_i <= right_borders[0]) :
            return 1
        elif (x>left_borders[1] and x <= right_borders[1])   and (x_i>left_borders[1] and x_i <= right_borders[1]):
            return 1
        elif (x>left_borders[2] and x <= right_borders[2])   and (x_i>left_borders[2] and x_i <= right_borders[2]):
            return 1
        elif (x>left_borders[3] and x <= right_borders[3])   and (x_i>left_borders[3] and x_i <= right_borders[3]):
            return 1
        elif (x>left_borders[4] and x <= right_borders[4])   and (x_i>left_borders[4] and x_i <= right_borders[4]):
            return 1
        elif (x>left_borders[5] and x <= right_borders[5])   and (x_i>left_borders[5] and x_i <= right_borders[5]):
            return 1
        elif (x>left_borders[6] and x <= right_borders[6])   and (x_i>left_borders[6] and x_i <= right_borders[6]):
            return 1
        elif (x>left_borders[7] and x <= right_borders[7])   and (x_i>left_borders[7] and x_i <= right_borders[7]):
            return 1
        elif (x>left_borders[8] and x <= right_borders[8])   and (x_i>left_borders[8] and x_i <= right_borders[8]):
            return 1
        elif (x>left_borders[9] and x <= right_borders[9])   and (x_i>left_borders[9] and x_i <= right_borders[9]):
            return 1
        else:
            return 0
        
        
    
    
    


# In[41]:


left_borders = np.arange(1.5, 5.2, h)
right_borders = np.arange(1.5 + h, 5.2 + h, h)
N = np.asarray([np.sum((left_borders[b] < x_train) & (x_train <= right_borders[b])) for b in range(len(left_borders))])
bin1=[]
bin2=[]
bin3=[]
bin4=[]
bin5=[]
bin6=[]
bin7=[]
bin8=[]
bin9=[]
bin10=[]
bin_list=[]
for i in range(0,150):
    if x_train[i]>left_borders[0] and x_train[i] <= right_borders[0]:
        bin1.append(y_train[i])
    elif x_train[i]>left_borders[1] and x_train[i] <= right_borders[1]:
        bin2.append(y_train[i])
    elif x_train[i]>left_borders[2] and x_train[i] <= right_borders[2]:
        bin3.append(y_train[i])
    elif x_train[i]>left_borders[3] and x_train[i] <= right_borders[3]:
        bin4.append(y_train[i])
    elif x_train[i]>left_borders[4] and x_train[i] <= right_borders[4]:
        bin5.append(y_train[i])
    elif x_train[i]>left_borders[5] and x_train[i] <= right_borders[5]:
        bin6.append(y_train[i])
    elif x_train[i]>left_borders[6] and x_train[i] <= right_borders[6]:
        bin7.append(y_train[i])
    elif x_train[i]>left_borders[7] and x_train[i] <= right_borders[7]:
        bin8.append(y_train[i])
    elif x_train[i]>left_borders[8] and x_train[i] <= right_borders[8]:
        bin9.append(y_train[i])
    else:
        bin10.append(y_train[i])

        

bin_list.append(bin1)
bin_list.append(bin2)
bin_list.append(bin3)
bin_list.append(bin4)
bin_list.append(bin5)
bin_list.append(bin6)
bin_list.append(bin7)
bin_list.append(bin8)
bin_list.append(bin9)
bin_list.append(bin10)


# In[42]:


list_REG=[]
for i in range(0,10):

    a= np.sum(np.asarray(bin_list[i]))/N[i]
    list_REG.append(a)



left_borders = np.arange(1.5, 5.2, h)
right_borders = np.arange(1.5 + h, 5.2 + h, h)

plt.figure(figsize = (10, 6))
plt.scatter(x_train,y_train,color="blue",label="training")
plt.scatter(x_test,y_test,color="red",label="test")
plt.ylabel("Waiting Time for Next Eruption (min)")
plt.xlabel("Eruption Time (min)")
plt.title("Regressogram Plot")
plt.legend()
for b in range(len(left_borders)):
    plt.plot([left_borders[b], right_borders[b]],[list_REG[b], list_REG[b]],  "k-")
for b in range(len(left_borders) - 1):
    plt.plot([right_borders[b], right_borders[b]], [list_REG[b], list_REG[b + 1]], "k-")    


# In[43]:


list_REG=[]
total=0
total_b=0
for n in range(0,len(x_test)):
    for i in range(0,len(x_train)):
        a=(B(x_test[n],x_train[i])*y_train[i])  
        b=B(x_test[n],x_train[i])
        total +=a
        total_b +=b
    list_REG.append(total/total_b)
        
    total=0
    total_b=0
RMSE=math.sqrt(np.sum((y_test-np.array(list_REG))**2)/122)



print("Regressogram => RMSE is" , RMSE , "when h is", h)


# In[44]:


sample_x = np.linspace(1.5, 5.2, 272)
trained_y = np.asarray([np.sum((((x - 0.5 * h) < x_train) & (x_train <= (x + 0.5 * h)))* y_train) / np.sum(((x - 0.5 * h) < x_train) & (x_train <= (x + 0.5 * h))) for x in sample_x]) 


# In[45]:


plt.figure(figsize = (10, 6))
plt.scatter(x_train,y_train,color="blue",label="training")
plt.scatter(x_test,y_test,color="red",label="test")
plt.ylabel("Waiting Time for Next Eruption (min)")
plt.xlabel("Eruption Time (min)")
plt.title("Running Mean Smoother Plot")
plt.plot(sample_x,trained_y,"k-")
plt.legend()
plt.show()


# In[46]:


list_RMS=[]
total=0
total_b=0
for n in range(0,len(x_test)):
    for i in range(0,len(x_train)):
        a=(W((x_test[n]-x_train[i])/h)*y_train[i])  
        b=W((x_test[n]-x_train[i])/h)
        total +=a
        total_b +=b
    list_RMS.append(total/total_b)
        
    total=0
    total_b=0
RMSE=math.sqrt(np.sum((y_test-np.array(list_RMS))**2)/122)
print("Running Mean Smoother => RMSE is" , RMSE , "when h is", h)


# In[47]:


def KernelSmoother(x,x_train,y_train):
    list_kernel=[]
    total=0
    total_b=0
    for n in range(0,len(x)):
        for i in range(0,len(x_train)):
            a=(K((x[n]-x_train[i])/h)*y_train[i])  
            b=K((x[n]-x_train[i])/h)
            total +=a
            total_b +=b
        list_kernel.append(total/total_b)
        total=0
        total_b=0
    
        
        
    return np.array(list_kernel)


# In[48]:


plt.figure(figsize = (10, 6))
plt.scatter(x_train,y_train,color="blue",label="training")
plt.scatter(x_test,y_test,color="red",label="test")
plt.ylabel("Waiting Time for Next Eruption (min)")
plt.xlabel("Eruption Time (min)")
plt.title("Kernel Smoother Plot")
x_kernel = np.linspace(1.5,5.2,272)
y_kernel = np.transpose(KernelSmoother(x_kernel,x_train,y_train))
plt.plot(x_kernel,y_kernel,color="black")
plt.legend()
plt.show()


# In[49]:


RMSE=math.sqrt(np.sum((y_test-np.transpose(KernelSmoother(x_test,x_train,y_train)))**2)/122)
print("Kernel Smoother => RMSE is" , RMSE , "when h is", h)


# In[ ]:





# In[ ]:




