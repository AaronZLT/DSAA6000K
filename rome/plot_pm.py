import os
import json
import copy
import numpy as np

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.ticker as ticker

import brewer2mpl
import random
import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as mcolors

import math
import pandas

def save(figure,str):
    figure.savefig(str+'.pdf')
    figure.savefig(str)
    
marker = ['.',',','o','v','^','<','>','1','2','3','4','s','p','*','h','H','+','x','D','d','|','_']
linestyle = ['-','--','-.',':']

popular_layer = []
popular_PM_layer = []
unpopular_layer = []
unpopular_PM_layer = []

def PM(file_path):
    with open(file_path) as f:
        data = json.load(f)

    popular_count = 0
    popular_PM_count = 0

    unpopular_count = 0
    unpopular_PM_count = 0

    for i in data:
        target_unpopular = i["requested_rewrite"]["target_unpopular"]
        generations_unpopular = i["generations_unpopular"]

        for j in range(len(target_unpopular)):
            unpopular_count = unpopular_count + len(generations_unpopular[j])
            unpopular_PM_count = unpopular_PM_count + find_string_in_list(target_unpopular[j],generations_unpopular[j])

    for i in data:
        target_new = i["requested_rewrite"]["target_new"]["str"]
        generations_popular = i["generations_popular"]
        popular_count = popular_count + len(generations_popular)
        popular_PM_count = popular_PM_count + find_string_in_list(target_new,generations_popular)
    
    popular_layer.append(popular_count)
    popular_PM_layer.append(popular_PM_count)
    unpopular_layer.append(unpopular_count)
    unpopular_PM_layer.append(unpopular_PM_count)

def find_string_in_list(main_string, string_list):
    result = 0
    main_string_lower = main_string.lower()
    contains_list = []

    for s in string_list:
        if main_string_lower in s.lower():
            result = result + 1
    
    return result

def find_max_value_and_index(lst):
    max_value = max(lst)
    max_index = lst.index(max_value)
    return max_value, max_index

def calculate_average_in_chunks(lst, chunk_size):
    averages = []
    
    for i in range(0, len(lst), chunk_size):
        chunk = lst[i:i + chunk_size]
        chunk_average = sum(chunk) / len(chunk)
        averages.append(chunk_average)
    
    return averages

for i in range(0,48):
    PM(f'save_layer_{i}.json')

print(popular_layer)
print(popular_PM_layer)
print(unpopular_layer)
print(unpopular_PM_layer)

popular = []
unpopular = []
for i in range(len(popular_layer)):
    popular.append(round(popular_PM_layer[i]/popular_layer[i], 2))
    unpopular.append(round(unpopular_PM_layer[i]/unpopular_layer[i], 3))

print()
print(popular)
print(np.mean(popular))
print(calculate_average_in_chunks(popular,10))

print(unpopular)
print(np.mean(unpopular))
print(calculate_average_in_chunks(unpopular,10))

fig, ax = plt.subplots(dpi=500,figsize=(6,4))
x=[i for i in range(0,48)]
x_tick=np.linspace(0,48,10)

ax.plot(x, popular, marker='s', linestyle = '-', lw=4, label='Popular')
ax.plot(x, unpopular, color = 'red', marker='+', linestyle = '-', lw=4, label='Unpopular')

m_p, i_p = find_max_value_and_index(popular)
m_up, i_up = find_max_value_and_index(unpopular)

ax.axhline(y=m_p, color='grey', linestyle='--',linewidth=0.5,xmin=0, xmax=i_p/48)
ax.axhline(y=m_up, color='grey', linestyle='--',linewidth=0.5,xmin=0, xmax=i_up/48)
ax.axvline(x=i_p, color='grey', linestyle='--',linewidth=0.5,ymin = 0, ymax = m_p)
ax.axvline(x=i_up, color='grey', linestyle='--',linewidth=0.5,ymin = 0, ymax = m_up)

plt.xlabel('Layers',fontsize=20)
plt.ylabel("PM (%)",fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
leg=ax.legend(loc='lower left',fontsize=15)
fig.tight_layout()
save(fig,"PM")