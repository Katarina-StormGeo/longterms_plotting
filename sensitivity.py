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

sensitivity_pth = 'M:/LongTerm/nena-lrmcmodel/results/Fall 2024/sensitivity.xlsx'
sensitivity = pd.read_excel(sensitivity_pth,header=4)
sensitivity.index = sensitivity.index + 6

#nuclear = sensitivity.loc[6:22]
p2p = sensitivity.loc[27:43]
#onshore = sensitivity.loc[49:65]
vrflow = sensitivity.loc[70:86]
liion = sensitivity.loc[92:108]
#offshore_btm = sensitivity.loc[113:129]
#offshore_flo = sensitivity.loc[134:150]
pem1 = sensitivity.loc[156:172]
#alkaline = sensitivity.loc[177:193]
alk_ded1 = sensitivity.loc[198:214]
alk_ded2 = sensitivity.loc[219:235]
#pem2 = sensitivity.loc[240:256]
#soec = sensitivity.loc[261:277]
ccs = sensitivity.loc[282:298]
rooftop_solar = sensitivity.loc[303:319]
#utility_solar = sensitivity.loc[324:340]


def plot_prep(tech):
    tech = tech.set_index('change')
    tech.index.name = None
    tech = tech. iloc[:, [3,7,11,15,19]]
    tech.index = ['-40%', '-35%', '-30%', '-25%', '-20%', '-15%', '-10%',
                  '-5%', '0%', '5%', '10%', '15%', '20%', '25%', '30%', '35%', '40%']
    tech.columns = ['capex', 'flh', 'lft','opex', 'wacc']

    return tech

techy = plot_prep(utility_solar)

fig, ax= plt.subplots(figsize=(8,7))
techy.plot(ax = ax,  linewidth = 2)
ax.legend(bbox_to_anchor =(0.5,-0.25),  loc='lower center', ncol=5,fontsize=12.5, frameon=False)


ax.xaxis.set_tick_params(labelsize=14)  
ax.yaxis.set_tick_params(labelsize=14)
ax.margins(x=0)
ax.grid(axis = 'y', alpha=0.3)
ax.set_ylabel('€/MWh',fontsize=15)
ax.set_ylim(0,110)


fig.savefig('../Figures_Renewables/SolarPower Fig 2. Sensitivity Analysis Utility Solar.pdf', bbox_inches='tight')

plt.show()


print(techy)