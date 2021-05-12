import pickle
import numpy as np

with open('OUTPUT/RealnessGAN-MixGauss-1_Extra/5/extra.pk', 'rb') as f:#'data/MixtureGaussian3By3.pk', 'rb') as f:
    new_data = pickle.load(f)


import matplotlib.pyplot as plt

arr = new_data[0]
x = arr[0]
y = arr[1]
s = 10
for i in range(len(new_data)-1):
    arr = new_data[i+1]
    x = np.append(x, arr[0])
    y = np.append(y, arr[1])
    s = np.append(s, 1)
    

print(len(new_data))

print(x)
print(y)
fig, ax = plt.subplots()
ax.scatter(x,y,s)
plt.show()


import math
mode_1 = 0 # (2,2)
mode_2 = 0 # (2,0)
mode_3 = 0 # (2,-2)
mode_4 = 0 # (0,2)
mode_5 = 0 # (0,0)
mode_6 = 0 # (0,-2)
mode_7 = 0 # (-2,2)
mode_8 = 0 # (-2,0)
mode_9 = 0 # (-2,-2)
recovered_modes = 0
mode_x = 0
mode_y = 0

for j in range(len(x)-1):
    mode_x = 2
    buff_x = x[j] - 2
    if(x[j] < 1 and x[j] > -1):
        mode_x = 0
        buff_x = x[j]
    if(x[j] < -1):
        mode_x = -2
        buff_x = x[j] + 2
    
    mode_y = 2
    buff_y = y[j] - 2
    if(y[j] < 1 and y[j] > -1):
        mode_y = 0
        buff_y = y[j]
    if(y[j] < -1):
        mode_y = -2
        buff_y = y[j] + 2
    print(buff_x)
    print(buff_y)
    print("---------------------")

    distance = math.sqrt(buff_x ** 2 + buff_y ** 2)

    if distance < 4 * 0.05: #if sample is high quality
        if mode_x == 2 and mode_y == 2:
            mode_1 = mode_1 + 1
            if mode_1 == 100:
                recovered_modes = recovered_modes + 1
        if mode_x == 2 and mode_y == 0:
            mode_2 = mode_2 + 1
            if mode_2 == 100:
                recovered_modes = recovered_modes + 1
        if mode_x == 2 and mode_y == -2:
            mode_3 = mode_3 + 1
            if mode_3 == 100:
                recovered_modes = recovered_modes + 1
        if mode_x == 0 and mode_y == 2:
            mode_4 = mode_4 + 1
            if mode_4 == 100:
                recovered_modes = recovered_modes + 1
        if mode_x == 0 and mode_y == 0:
            mode_5 = mode_5 + 1
            if mode_5 == 100:
                recovered_modes = recovered_modes + 1
        if mode_x == 0 and mode_y == -2:
            mode_6 = mode_6 + 1
            if mode_6 == 100:
                recovered_modes = recovered_modes + 1
        if mode_x == -2 and mode_y == 2:
            mode_7 = mode_7 + 1
            if mode_7 == 100:
                recovered_modes = recovered_modes + 1
        if mode_x == -2 and mode_y == 0:
            mode_8 = mode_8 + 1
            if mode_8 == 100:
                recovered_modes = recovered_modes + 1
        if mode_x == -2 and mode_y == -2:
            mode_9 = mode_9 + 1
            if mode_9 == 100:
                recovered_modes = recovered_modes + 1
high_quality = (mode_1 + mode_2 + mode_3 + mode_4 + mode_5 + mode_6 + mode_7 + mode_8 + mode_9) * 100 / len(x)
print(high_quality)
print(recovered_modes)
print("-----------------------------")
print(mode_1)
print(mode_2)
print(mode_3)
print(mode_4)
print(mode_5)
print(mode_6)
print(mode_7)
print(mode_8)
print(mode_9)











