import pandas as pd
import matplotlib.pyplot as plt
import math
from sympy import *     
import numpy as np
from scipy.interpolate import interp1d

df = pd.read_csv("anime_df.csv")
theta = 0
pa = 5
total = []

first_row = df.iloc[0]
value_counts = first_row[(first_row != 0) & (first_row.index != 'anime_id')].value_counts()
x = value_counts.index
y = value_counts.values
total_non_zero = value_counts.sum()
probability_distribution = value_counts / total_non_zero
newpro = probability_distribution.sort_index()
ax = (newpro.index).tolist()
ay = (newpro.values).tolist()
# plt.bar(probability_distribution.index, probability_distribution.values)
# plt.xlabel('Value')
# plt.ylabel('Probability')
# plt.title('Probability Distribution of Values')
# plt.show()
expected_value = sum(probability_distribution.index * probability_distribution.values)
print(expected_value)
total.append([df['anime_id'][0],pa*10,expected_value,0])

# 使用线性插值方法创建连续的概率函数
interp_func = interp1d(ax, ay, kind='linear')
# 生成连续的概率函数值
ax_values = np.arange(min(ax), max(ax), 0.1)  # 这里以0.1为间隔生成5到10的数据
continuous_probs = interp_func(ax_values)
ax = ax_values.tolist()
normalized_probs = continuous_probs / np.sum(continuous_probs)
ay = (normalized_probs).tolist()
cul_ay = ay
for i in range(1,len(cul_ay),1):
    cul_ay[i] = cul_ay[i] + cul_ay[i-1]
    

for ad in range(1,len(df),1):
    first_row = df.iloc[ad]
    value_counts = first_row[(first_row != 0) & (first_row.index != 'anime_id')].value_counts()
    x = value_counts.index
    y = value_counts.values
    total_non_zero = value_counts.sum()
    probability_distribution = value_counts / total_non_zero
    newpro = probability_distribution.sort_index()
    bx = (newpro.index).tolist()
    by = (newpro.values).tolist()
    # plt.bar(probability_distribution.index, probability_distribution.values)
    # plt.xlabel('Value')
    # plt.ylabel('Probability')
    # plt.title('Probability Distribution of Values')
    # plt.show()
    expected_value = sum(probability_distribution.index * probability_distribution.values)
    
    
    # 使用线性插值方法创建连续的概率函数
    interp_func = interp1d(bx, by, kind='linear')
    # 生成连续的概率函数值
    bx_values = np.arange(min(bx), max(bx), 0.1)  # 这里以0.1为间隔生成5到10的数据
    continuous_probs = interp_func(bx_values)
    bx = bx_values.tolist()
    normalized_probs = continuous_probs / np.sum(continuous_probs)
    by = (normalized_probs).tolist()
    cul_by = by
    for i in range(1,len(cul_by),1):
        cul_by[i] = cul_by[i] + cul_by[i-1]
    
    
    for a in range(0,len(ax),1):
        if ax[a] >pa:
            FPA = cul_ay[a-1]
            break
    
    derivative = 100   
    final_pb = 0     
    for x in range(0,len(bx),1):
        temp = 0
        for a in ax:
            if a >= pa:
                temp = temp + ay[ax.index(a)]*cul_by[x]
        ans = (1-(cul_by[x]*FPA) - temp)
        if abs(ans-0)<abs(derivative-0):
            derivative = ans
            final_pb = bx[x]
    #print("result:",ad, final_pb,derivative)
    
    total.append([df['anime_id'][ad], final_pb*10, expected_value, derivative])


total_df = pd.DataFrame(total, columns=['anime_id', 'price', 'expected preference', 'derivative'])
total_df.to_csv("anime_with_price.csv")
