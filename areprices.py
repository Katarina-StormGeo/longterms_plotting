import json
import pandas as pd
import xarray as xr
import yaml
import time
from itertools import dropwhile
import sys 
import os
import subprocess
import requests
import argparse
from io import StringIO
import logging
from logging.handlers import TimedRotatingFileHandler
import numpy as np
import traceback
import operator
import re
from dateutil.relativedelta import relativedelta
import glob
from datetime import datetime, timedelta, date
from calendar import monthrange, month_name
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

nordic_power = r'M:\Prosjekter\Long Term Nordic\2024 Fall\Nordic Power Market Outlook - 2024-2050 (Fall 24) LINKS.xlsx'

area_prices = pd.read_excel(nordic_power, sheet_name='Area prices Base')
area_prices = area_prices.drop(columns=['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4','Unnamed: 5',])
epads = area_prices[15:27]
epads = epads.set_index('€/MWh')
epads.index.name = None
epads = epads.transpose()


fig, ax= plt.subplots(figsize=(8,7))

colors = ['#FF6F91', '#2E4053', '#F39C12', '#8E44AD', '#3498DB', '#16A085', '#E74C3C', 'gray', '#F1C40F', 'navy', 'turquoise', 'darkgreen']




epads.plot(ax = ax,  linewidth = 3, color=colors)
ax.legend(bbox_to_anchor =(0.5,-0.35),  loc='lower center', ncol=4,fontsize=15, frameon=False)

ax.set_xticks(range(len(epads.index)))
ax.xaxis.set_tick_params(labelsize=17, rotation = 90)  
ax.yaxis.set_tick_params(labelsize=17)
ax.margins(x=0)
ax.grid(axis = 'y', alpha=0.3)
ax.set_ylabel('€/MWh',fontsize=17)
ax.set_ylim(-30,40)

fig.savefig('../Figurer_presentasjon/The Nordics Fig 7.  EPADs (Electricity Price Area Differentials.png', bbox_inches='tight')

#plt.show()

sysprice = area_prices.loc[0]
range = area_prices[1:13]
range = range.set_index('€/MWh')

range.index.name = None

mini = range.min()
maxi = range.max()


sysi = sysprice.values[1:]
fig, ax= plt.subplots(figsize=(8,7))
ax.fill_between(mini.index.values.astype(float),maxi.values.astype(float), mini.values.astype(float),alpha = 0.4)
ax.plot(mini.index.values.astype(float), sysi,label = 'Sys Price', color = 'firebrick', linewidth = 4)

ax.legend(bbox_to_anchor =(0.5,-0.25),  loc='lower center', ncol=1,fontsize=17, frameon=False)

ax.xaxis.set_tick_params(labelsize=17, rotation = 90)  
ax.yaxis.set_tick_params(labelsize=17)
ax.margins(x=0)
ax.grid(axis = 'y', alpha=0.3)
ax.set_ylabel('€/MWh',fontsize=17)
ax.set_ylim(20,90)


fig.savefig('../Figurer_presentasjon/The Nordics Fig 8.  Area Price Range.png', bbox_inches='tight')


