
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


input_cost = r'M:\Prosjekter\Long Term Nordic\2024 Fall\Input costs Nordic Fall 2024.xlsx'
results = r'M:\LongTerm\nena-lrmcmodel\results\Fall 2024\results.xlsx'
nordic_growth_pth = r'M:\Prosjekter\Green Certificates\2015 October\Forward exchange rates_Longterms_Fall_2015.xlsx'
nordic_power = r'M:\Prosjekter\Long Term Nordic\2024 Fall\Nordic Power Market Outlook - 2024-2050 (Fall 24) LINKS.xlsx'

nuc_cap = r'M:\Prosjekter\Long Term Nordic\2016 fall\Tables and figures.xlsx'
german_pp = r'M:\EUROPA\Germany\German power plants and installed capacity.xlsx'

base = pd.read_excel(input_cost, sheet_name = 'Base Scenario')
base.columns = base.columns.astype(str)
lrmc_df = base.drop(columns=['Unnamed: 1', 'Unnamed: 2', '2023', '2024', 'Unnamed: 31'])


summary = pd.read_excel(german_pp, sheet_name = 'LT misc', header = 1)
summary.columns = summary.columns.astype(str)
summary = summary[['€/MWh', '2025', '2030', '2035', '2040', '2050']].copy()
summary.index = summary.index + 3


val_results = pd.read_excel(results, header=2)
val_results.columns = val_results.columns.astype(str)
val_results = val_results.rename(columns={'Unnamed: 0': 'params'})
val_results = val_results.drop(columns=['Unnamed: 1', 'Unnamed: 2', '2024'])

def return_xticks(all_years, specific_years):
    new_xticks = []
    for i in all_years:
        if i in specific_years:
            new_xticks.append(i)
        else:
            new_xticks.append('')
    return new_xticks
    



def summary_tables(table_name, indices, count):
    table = summary.loc[indices]
    table = table.set_index('€/MWh')
    table.index.name = None
    table = table.transpose()

    if table_name in ['prices', 'price_outcome', 'EPAD_extract', 'SRMCs_and_EUAs']:
        y = '€/MWh'
    else:
        y = 'TWh'

    fig, ax = plt.subplots(figsize = (8,7))

    table.plot(ax = ax, linewidth = 2)
    n = 2
    if table_name == 'SRMCs_and_EUAs':
        n = 3
    ax.legend(bbox_to_anchor =(0.5,-0.3),  loc='lower center', ncol=n,fontsize=13, frameon=False)

    ax.set_xticks(range(len(table.index)))
    ax.xaxis.set_tick_params(labelsize=14,rotation=90)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.margins(x=0)
    ax.grid(axis = 'y', alpha=0.3)
    ax.set_ylabel('{}'.format(y),fontsize=15)

    if table_name == 'net_exchange':
        ax.set_ylim(-60,20)

    name = '{}'.format(table_name.replace('_', ' ')).capitalize()
    fig.savefig('../Figures_Power/Executive Summary fig {} {}.pdf'.format(count, name), bbox_inches='tight')

    return table


summaries = {'prices': [3,4,5,6,7], 'price_outcome': [15,16,17,18], 'EPAD_extract': [27,28,29,30,31,32],\
             'SRMCs_and_EUAs': [39,40,41], 'consumption': [52,53,54,55], 'wind': [64,65,66,67],\
             'solar': [76,77,78,79], 'net_exchange': [88,89,90,91]}
count = 1
for i in summaries:
    summary_tables(i, summaries[i], count)
    count +=1
    #print(summaries[i])
#plt.show()


########################################################################################
"""
consumption = pd.read_excel(nordic_power,sheet_name = 'Consumption',header=52)
consumption.columns = consumption.columns.astype(str)
consumption = consumption.drop(columns=['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5'])
consumption = consumption.set_index('Nordic')
consumption = consumption.transpose()

print(consumption)

hydrogen = pd.DataFrame({'Hydrogen': consumption['Hydrogen'] - consumption['Hydrogen'].iloc[0]})
el_vehicles = pd.DataFrame({'Electric vehicles': consumption['Electric vehicles'] - consumption['Electric vehicles'].iloc[0]})
data_storage = pd.DataFrame({'Data Storage': consumption['Datastorage'] - consumption['Datastorage'].iloc[0]})
industry = pd.DataFrame({'Industry': consumption['Industry excl. H2'] - consumption['Industry excl. H2'].iloc[0]})
losses = pd.DataFrame({'Losses incl. Pumped Storage': consumption['Losses incl Pumped storage'] - consumption['Losses incl Pumped storage'].iloc[0]})
petroleum = pd.DataFrame({'Petroleum Sector': consumption['Petroleum sector'] - consumption['Petroleum sector'].iloc[0]})
                      
cons = pd.concat([hydrogen, el_vehicles, data_storage, industry, losses, petroleum], axis=1)

colors = ['#005ca4', '#DDEEFA', '#BDE0FE', '#89CFF0', '#6CAEED', '#013863']

fig, ax = plt.subplots(figsize=(8,7))
labels = ['Hydrogen', 'Electric vehicles', 'Data Storage', 'Industry', 'Losses', 'Petroleum Sector']
ax.stackplot(cons.index, [cons['Hydrogen'], cons['Electric vehicles'],
                           cons['Data Storage'], cons['Industry'],
                           cons['Losses incl. Pumped Storage'], cons['Petroleum Sector']], colors = colors, labels = labels)


ax.legend(bbox_to_anchor =(0.5,-0.35), loc='lower center', ncol=2,fontsize=14, frameon=False)
xticks = return_xticks(cons.index, ['2025','2030','2035','2040','2045','2050'])
ax.set_xticklabels(xticks,rotation = 90,fontsize=14)
ax.yaxis.set_tick_params(labelsize=14)
ax.margins(x=0)
ax.grid(axis = 'y', alpha=0.3)
ax.set_ylabel('TWh',fontsize=15)

fig.savefig('../Figures_Power/The Nordics fig 1. Electricity consumption growth 2025-2050.pdf', bbox_inches='tight')
fig.savefig('../Figures_Power/Nordic Power Base Scenario Fig 1. Main drivers for consumption growth.pdf', bbox_inches='tight')



plt.show()
"""
#####################################################################################
"""
nordic_growth = pd.read_excel(nordic_growth_pth,sheet_name='Nordics',header=1)
nordic_growth = nordic_growth[37:48]
nordic_growth.columns = nordic_growth.columns.astype(str)
nordic_growth = nordic_growth.rename(columns={'Investments (TWh)': 'params'})
nordic_growth = nordic_growth.drop(columns=['2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024'])



params = ['Wind onshore', 'Wind offshore', 'Hydro', 'CHP', 'Solar Utility scale', 'Solar Rooftop', 'Nuclear']

indices = []
for i in params:
    index = nordic_growth.index[nordic_growth['params'] == i].values[0]
    indices.append(index)


nordic = nordic_growth.loc[indices]
nordic = nordic.set_index('params')
nordic.index.name = None
nordic = nordic.transpose()

x = nordic.index
y = [nordic['Wind onshore'], nordic['Wind offshore'], nordic['Hydro'], nordic['CHP'], nordic['Solar Utility scale'],\
     nordic['Solar Rooftop']]
colors = ['lightsteelblue', 'steelblue', 'navy', '#7aa741', '#f4d65c', '#e3ad12']

fig, ax = plt.subplots(figsize  = (8,7))


ax.stackplot(x,y, colors = colors, alpha = 0.85, labels = nordic.columns)
ax.legend(bbox_to_anchor =(0.5,-0.35), loc='lower center', ncol=3,fontsize=14, frameon=False)
xticks = return_xticks(nordic.index, ['2025','2030','2035','2040','2045','2050'])
ax.set_xticklabels(xticks,rotation = 90,fontsize=14)
ax.yaxis.set_tick_params(labelsize=14)
ax.margins(x=0)
ax.grid(axis = 'y', alpha=0.3)
ax.set_ylabel('TWh',fontsize=15)

fig.savefig('../Figures_Power/Nordic Power Base Scenario Fig 3. Nordic Renewable Growth.pdf', bbox_inches='tight')

plt.show()
"""
####################################################################################

"""
net_exchange = pd.read_excel(nordic_power, sheet_name='Base case')
net_exchange = net_exchange.drop(columns=['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5'])
net_exchange.index = net_exchange.index + 2

net_exchange = net_exchange.loc[[34]]
net_exchange = net_exchange.set_index('€/MWh')
net_exchange.index.name = None

net_exchange = net_exchange.transpose()
net_exchange = net_exchange[1:]




fig, ax= plt.subplots(figsize=(8,7))
net_exchange.plot(ax = ax, color = "darkorange", linewidth = 2)
ax.legend(bbox_to_anchor =(0.5,-0.25),  loc='lower center', ncol=1,fontsize=12.5, frameon=False)

ax.set_xticks(range(len(net_exchange.index)))
ax.xaxis.set_tick_params(labelsize=14,rotation=90)  
ax.yaxis.set_tick_params(labelsize=14)
ax.margins(x=0)
ax.grid(axis = 'y', alpha=0.3)
ax.set_ylabel('TWh',fontsize=15)
ax.set_ylim(-100,-20)

fig.savefig('../Figures_Power/Nordic Power Base Scenario Fig 2. Net Exchange.pdf', bbox_inches='tight')
plt.show()
"""

###################################################################################

"""
hourly_prices = r'M:\Prosjekter\Long Term Nordic\2024 Fall\Hourly prices Base Fall 2024.xlsx'
average = r'M:\Prosjekter\Green Certificates\2024 October\Katarina\2001-2020 avg prices.xlsx'
average = pd.read_excel(average, header = 2)
average = average[['Sys avg. 2001-2020']][2:].reset_index().drop(columns=['index'])
average = average['Sys avg. 2001-2020'].sort_values().values

hourly_prices = pd.read_excel(hourly_prices, sheet_name = 'SYS')
hourly_prices.columns = hourly_prices.columns.astype(str)
hourly_prices = hourly_prices.drop(columns=['Unnamed: 0', 'Unnamed: 1'])
hourly_prices = hourly_prices[2:]
hourly_prices = hourly_prices.reset_index().drop(columns=['index'])

y25 = hourly_prices['2025'].sort_values().values
y40 = hourly_prices['2040'].sort_values().values
y50 = hourly_prices['2050'].sort_values().values


percentile = {'percentage': [], '2025': [], '2040': [], '2050': [], 'Avg 2001-2020': []}
for i in range(0,101):
    y25_perc = np.percentile(y25, 100-i)
    y40_perc = np.percentile(y40, 100-i)
    y50_perc = np.percentile(y50, 100-i)
    avg = np.percentile(average, 100-i)

    percentile['percentage'].append('{} %'.format(i))
    percentile['2025'].append(y25_perc)
    percentile['2040'].append(y40_perc)
    percentile['2050'].append(y50_perc)
    percentile['Avg 2001-2020'].append(avg)

percentiles = pd.DataFrame.from_dict(percentile)
percentiles = percentiles.set_index('percentage')
percentiles.index.name = None


fig, ax= plt.subplots(figsize=(8,7))
percentiles.plot(ax = ax, linewidth = 2)
ax.legend(bbox_to_anchor =(0.5,-0.25),  loc='lower center', ncol=4,fontsize=12.5, frameon=False)

#ax.set_xticks(range(len(percentiles.index)))
ax.xaxis.set_tick_params(labelsize=14,rotation=0)  
ax.yaxis.set_tick_params(labelsize=14)
ax.margins(x=0)
ax.grid(axis = 'y', alpha=0.3)
ax.set_ylabel('€/MWh',fontsize=15)

fig.savefig('../Figures_Power/Nordic Power Base Scenario Fig 4. System price distribution.pdf', bbox_inches='tight')
plt.show()
"""


##################################################
"""
nordics_lrmc_vars = ['Onshore wind Nordics', 'Offshore wind Nordic West (bottom fixed)', 'Offshore wind Nordic West (floating)', 'PV (Roof top) Nordics', 'PV (Utility scale) Nordics', 'Bio CHP']


indices = []
for i in nordics_lrmc_vars:
    index = base.index[base['params'] == i].values[0]
    indices.append(index)


nordics_lrmc_df = lrmc_df.loc[indices]
nordics_lrmc_df = nordics_lrmc_df.set_index('params')
nordics_lrmc_df.index.name = None
nordics_lrmc_df = nordics_lrmc_df.transpose()


fig2, ax1 = plt.subplots(figsize = (8,7))


nordics_lrmc_df['Onshore wind Nordics'].plot(ax = ax1, 
                                             color = 'skyblue', 
                                             linewidth= 2,
                                             label = 'LRMC Onshore wind Nordics (3410 FLH,WACC 6.2%)')
nordics_lrmc_df['Offshore wind Nordic West (bottom fixed)'].plot(ax = ax1, dashes=[6, 2],
                                                                 linewidth = 2,
                                                                 color = 'steelblue',
                                                                 label = 'LRMC Offshore wind Nordic West (bottom fixed) (4509 FLH, WACC 7.5%)')
nordics_lrmc_df['Offshore wind Nordic West (floating)'].plot(ax=ax1,
                                                             color = 'steelblue',
                                                             linewidth = 2,
                                                             label = 'LRMC Offshore wind Nordic West (floating) (4509 FLH, WACC 7.5%)')
nordics_lrmc_df['PV (Roof top) Nordics'].plot(ax = ax1,
                                              color = 'orange',
                                              linewidth = 2,
                                              label = 'LRMC PV (Roof top) Nordics (959 FLH, WACC 5.7%)')
nordics_lrmc_df['PV (Utility scale) Nordics'].plot(ax = ax1, dashes = [6,2],
                                                   color = 'orange',
                                                   linewidth = 2,
                                                   label = 'LRMC PV (Utility scale) Nordics (1020 FLH, WACC 6.2%)')
nordics_lrmc_df['Bio CHP'].plot(ax = ax1,
                                color = 'olivedrab',
                                linewidth = 2,
                                label = 'Bio CHP (wood chips) with 25 MWel (5500 FLH, WACC 7.5%)')


fig2.legend(bbox_to_anchor =(0.5,-0.25),  loc='lower center', ncol=1,fontsize=12.5, frameon=False)

ax1.set_xticks(range(len(nordics_lrmc_df.index)))
ax1.xaxis.set_tick_params(labelsize=14,rotation=90)
ax1.yaxis.set_tick_params(labelsize=14)
ax1.margins(x=0)
ax1.grid(axis = 'y', alpha=0.3)
ax1.set_ylabel('€/MWh',fontsize=15)


fig2.savefig('../Figures_Power/Nordic Power Base Scenario Fig 5. LRMCs.pdf', bbox_inches='tight')

plt.show()
"""
###################################################################
"""
weather = pd.read_excel(nordic_power, sheet_name='Weather sensitivity Base case')
weather = weather.drop(columns=['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5'])
weather.columns = weather.columns.astype(str)
weather = weather.set_index('€/MWh')
weather.index.name = None
weather = weather.transpose()


fig, ax= plt.subplots(figsize=(8,7))
weather.plot(ax=ax,color = ['firebrick','mediumblue', 'gold'], linewidth=2)
ax.legend(bbox_to_anchor =(0.5,-0.25),  loc='lower center', ncol=4,fontsize=12.5, frameon=False)

#ax.set_xticks(range(len(percentiles.index)))
ax.xaxis.set_tick_params(labelsize=14,rotation=0)  
ax.yaxis.set_tick_params(labelsize=14)
ax.margins(x=0)
ax.grid(axis = 'y', alpha=0.3)
ax.set_ylabel('€/MWh',fontsize=15)

fig.savefig('../Figures_Power/Nordic Power Base Scenario Fig 6. Weather Sensitivity.pdf', bbox_inches='tight')

plt.show()
"""

####################################################################



"""
GoO = lrmc_df.iloc[[4]]
GoO = GoO.set_index('params')
GoO.index.name = None
GoO = GoO.transpose()
GoO.columns = ['Renewable Guarantee of Origin price'] 
print(GoO)

#plt.figure()

fig, ax= plt.subplots(figsize=(8,7))
GoO.plot(ax = ax, color = "olivedrab", linewidth = 2)
ax.legend(bbox_to_anchor =(0.5,-0.25),  loc='lower center', ncol=1,fontsize=12.5, frameon=False)

ax.set_xticks(range(len(GoO.index)))
ax.xaxis.set_tick_params(labelsize=14,rotation=90)
ax.yaxis.set_tick_params(labelsize=14)
ax.margins(x=0)
ax.grid(axis = 'y', alpha=0.3)
ax.set_ylabel('€/MWh',fontsize=15)
ax.set_yticks(range(0,4))
fig.savefig('../Figures_Power/Key enabling technologies for the green transition Fig 1. Renewable GoOs.pdf', bbox_inches='tight')
plt.show()

"""
#################################################################
"""
nordics_lrmc_vars = ['Onshore wind Norway', 
                     'Onshore wind Nordics', 
                     'Offshore wind Nordic West (bottom fixed)', 
                     'Offshore wind Nordic East (bottom fixed) ',
                     'Offshore wind Nordic West (floating)', 
                     'Offshore wind Nordic East (floating)',
                     'PV (Roof top) Nordics', 
                     'PV (Utility scale) Nordics', 
                     'Bio CHP']

print(base)
indices = []
for i in nordics_lrmc_vars:
    index = base.index[base['params'] == i].values[0]
    print(index)
    indices.append(index)


nordics_lrmc_df = lrmc_df.loc[indices]
nordics_lrmc_df = nordics_lrmc_df.set_index('params')
nordics_lrmc_df.index.name = None
nordics_lrmc_df = nordics_lrmc_df.transpose()


fig2, ax1 = plt.subplots(figsize = (8,7))

nordics_lrmc_df['Onshore wind Norway'].plot(ax = ax1, dashes = [6,2],
                                             color = 'forestgreen', 
                                             linewidth= 2,
                                             label = 'LRMC Onshore wind Norway (3511 FLH,WACC 6.2%)')
nordics_lrmc_df['Onshore wind Nordics'].plot(ax = ax1, 
                                             color = 'forestgreen', 
                                             linewidth= 2,
                                             label = 'LRMC Onshore wind Nordics (3410 FLH,WACC 6.2%)')

nordics_lrmc_df['Offshore wind Nordic West (bottom fixed)'].plot(ax = ax1, dashes=[6, 2],
                                                                 linewidth = 2,
                                                                 color = 'steelblue',
                                                                 label = 'LRMC Offshore wind Nordic West (bottom fixed) (4509 FLH, WACC 7.5%)')
nordics_lrmc_df['Offshore wind Nordic West (floating)'].plot(ax=ax1,
                                                             color = 'steelblue',
                                                             linewidth = 2,
                                                             label = 'LRMC Offshore wind Nordic West (floating) (4509 FLH, WACC 7.5%)')

nordics_lrmc_df['Offshore wind Nordic East (bottom fixed) '].plot(ax = ax1, dashes=[6, 2],
                                                                 linewidth = 2,
                                                                 color = 'navy',
                                                                 label = 'LRMC Offshore wind Nordic East (bottom fixed) (4158 FLH, WACC 7.5%)')
nordics_lrmc_df['Offshore wind Nordic East (floating)'].plot(ax=ax1,
                                                             color = 'navy',
                                                             linewidth = 2,
                                                             label = 'LRMC Offshore wind Nordic East (floating) (4158 FLH, WACC 7.5%)')

nordics_lrmc_df['PV (Roof top) Nordics'].plot(ax = ax1,
                                              color = 'orange',
                                              linewidth = 2,
                                              label = 'LRMC PV (Roof top) Nordics (959 FLH, WACC 5.7%)')
nordics_lrmc_df['PV (Utility scale) Nordics'].plot(ax = ax1, dashes = [6,2],
                                                   color = 'orange',
                                                   linewidth = 2,
                                                   label = 'LRMC PV (Utility scale) Nordics (1020 FLH, WACC 6.2%)')
nordics_lrmc_df['Bio CHP'].plot(ax = ax1,
                                color = 'saddlebrown',
                                linewidth = 2,
                                label = 'Bio CHP (wood chips) with 25 MWel (5500 FLH, WACC 7.5%)')


fig2.legend(bbox_to_anchor =(0.5,-0.35),  loc='lower center', ncol=1,fontsize=12.5, frameon=False)

ax1.set_xticks(range(len(nordics_lrmc_df.index)))
ax1.xaxis.set_tick_params(labelsize=14,rotation=90)
ax1.yaxis.set_tick_params(labelsize=14)
ax1.margins(x=0)
ax1.grid(axis = 'y', alpha=0.3)
ax1.set_ylabel('€/MWh',fontsize=15)


fig2.savefig('../Figures_Power/Key enabling technologies for the green transition Fig 2. Renewable LRMCs.pdf', bbox_inches='tight')

plt.show()
"""
###################################################################

"""

nordics_lrmc_vars = ['PEMEL Flexible (5 €/MWh input)', 'PEMEL Flat (50 €/MWh input)',
                     'PEMEL Flexible (20 €/MWh input)', 'PEMEL Flat (70 €/MWh input)',
                     'LRMC Grey H2 €/MWh', 'LRMC Blue H2 €/MWh']


indices = []
for i in nordics_lrmc_vars:
    index = base.index[base['params'] == i].values[0]
    indices.append(index)


hydrogen_lrmc_df = lrmc_df.loc[indices]
hydrogen_lrmc_df = hydrogen_lrmc_df.set_index('params')
hydrogen_lrmc_df.index.name = None
hydrogen_lrmc_df = hydrogen_lrmc_df.transpose()

hdry = hydrogen_lrmc_df[['PEMEL Flexible (5 €/MWh input)', 'PEMEL Flexible (20 €/MWh input)']].copy()
hdry2 = hydrogen_lrmc_df[['PEMEL Flat (50 €/MWh input)', 'PEMEL Flat (70 €/MWh input)','LRMC Grey H2 €/MWh', 'LRMC Blue H2 €/MWh']].copy()

fig, ax = plt.subplots(figsize=(8,7))

hdry.plot(ax = ax, color = ['#9cc674', 'darkgreen'], linewidth = 2, dashes=[6,2])
hdry2.plot(ax = ax, color = ['#9cc674', 'darkgreen', 'grey', '#0264b1'], linewidth = 2)

ax.legend(bbox_to_anchor =(0.5,-0.45), loc='lower center', ncol=1,fontsize=13, frameon=False)
ax.set_xticks(range(len(hygen_df.index)))
ax.xaxis.set_tick_params(labelsize=14,rotation=90)
ax.yaxis.set_tick_params(labelsize=14)
ax.margins(x=0)
ax.grid(axis = 'y', alpha=0.3)
ax.set_ylabel('€/MWh',fontsize=15)

fig.savefig('../Figures_Power/Key enabling technologies for the green transition Fig 3.  LRMCs H2 (fuel and CO2 price from Base scenario).pdf', bbox_inches='tight')

plt.show()
"""
#################################################################


"""
ttf_co2 = lrmc_df.iloc[[0, 2]]
ttf_co2 = ttf_co2.set_index('params')
ttf_co2.index.name = None
ttf_co2 = ttf_co2.transpose()
ttf_co2.columns = ['Natural Gas (TTF)', 'CO2 (EUA)']

fig,ax = plt.subplots(figsize=(8,7))

ttf_co2.plot(ax = ax, color= ['firebrick', 'darkgreen'], linewidth = 2)
ax.legend(bbox_to_anchor =(0.5,-0.25), loc='lower center', ncol=1,fontsize=13, frameon=False)
ax.set_xticks(range(len(ttf_co2.index)))
ax.xaxis.set_tick_params(labelsize=14,rotation=90)
ax.yaxis.set_tick_params(labelsize=14)
ax.margins(x=0)
ax.grid(axis = 'y', alpha=0.3)
ax.set_ylabel('€/MWh',fontsize=15)
fig.savefig('../Figures_Power/Key enabling technologies for the green transition Fig 4.  Natural gas (TTF) and CO2 (EUA) prices.pdf', bbox_inches='tight')

plt.show()
"""

################################################################
"""

hygen_price = lrmc_df.iloc[[11]]
hygen_price = hygen_price.set_index('params')
hygen_price.index.name = None
hygen_price = hygen_price.transpose()
hygen_price.columns = ['H2 price €/MWh']

fig, ax= plt.subplots(figsize=(8,7))
hygen_price.plot(ax = ax, color = "steelblue", linewidth = 2)
ax.legend(bbox_to_anchor =(0.5,-0.25),  loc='lower center', ncol=1,fontsize=12.5, frameon=False)

ax.set_xticks(range(len(hygen_price.index)))
ax.xaxis.set_tick_params(labelsize=14,rotation=90)  
ax.yaxis.set_tick_params(labelsize=14)
ax.margins(x=0)
ax.grid(axis = 'y', alpha=0.3)
ax.set_ylabel('€/MWh',fontsize=15)

fig.savefig('../Figures_Power/Key enabling technologies for the green transition Fig 5.  Modelled input H2 price for H2-fired generation.pdf', bbox_inches='tight')



plt.show()
"""




###############################################################
"""

flex_lrmc = lrmc_df.iloc[[43, 44, 41, 47, 55, 56]]

flex_lrmc = flex_lrmc.set_index('params')
flex_lrmc.index.name = None
flex_lrmc = flex_lrmc.transpose()

flex_lrmc = flex_lrmc.rename(columns={'H2GT (35 % Eff.)': 'H2GT (3000 FLH, WACC 7.5%)',\
                                      'FUEL CELL (60.6 % Eff.)': 'FUEL CELL (3000 FLH, WACC 7,5 %)',\
                                      'H2CCGT (58 % Eff.)': 'H2CCGT (8322 FLH, WACC 7,5%)',\
                                      'PtP (10 €/MWh input)': 'PtP (10 €/MWh input) (3000 FLH, WACC 7,5%)',\
                                      '20h VRF Battery (Weekly) (10€/MWh)':'20h VRF Battery (Weekly) (10€/MWh) (2555 FLH, WACC 7,5%)',\
                                      '4h Li-Ion Battery (Intraday) (10€/MWh)':'4h Li-Ion Battery (Intraday) (10€/MWh) (2920 FLH, WACC 7,5%)'})

#flex_lrmc.plot()

fig4,ax2 = plt.subplots(figsize = (8,7))

flex_lrmc.plot(ax = ax2, linewidth=2)
ax2.legend(bbox_to_anchor =(0.5,-0.45), loc='lower center', ncol=1,fontsize=13, frameon=False)
ax2.set_xticks(range(len(flex_lrmc.index)))
ax2.xaxis.set_tick_params(labelsize=14,rotation=90)
ax2.yaxis.set_tick_params(labelsize=14)
ax2.margins(x=0)
ax2.grid(axis = 'y', alpha=0.3)
ax2.set_ylabel('€/MWh',fontsize=15)


fig4.savefig('../Figures_Power/Key enabling technologies for the green transition Fig 6.  LRMCs - Energy storage and H2-fired gas generation.pdf', bbox_inches='tight')

plt.show()

"""
###############################################################
"""
nuclear_lrmc = lrmc_df.iloc[[85,86]]
nuclear_lrmc = nuclear_lrmc.set_index('params')
nuclear_lrmc.index.name = None
nuclear_lrmc = nuclear_lrmc.transpose()

nuclear_lrmc = nuclear_lrmc.rename(columns={'New Nuclear': 'New Nuclear (8059 FLH, WACC 8,5%)',\
                                      'Nuclear LTO': 'Nuclear LTO (7446 FLH, WACC 7,5 %)'})

new_nuclear = nuclear_lrmc[['New Nuclear (8059 FLH, WACC 8,5%)']].copy()
nuclear_lto = nuclear_lrmc[['Nuclear LTO (7446 FLH, WACC 7,5 %)']].copy()

fig, ax = plt.subplots(figsize=(8,7))

new_nuclear.plot(ax = ax, color = 'darkslateblue', linewidth = 2)
nuclear_lto.plot(ax = ax, color = 'darkslateblue', linewidth = 2, dashes = [6, 2])


ax.legend(bbox_to_anchor =(0.5,-0.25),  loc='lower center', ncol=1,fontsize=12.5, frameon=False)

ax.set_xticks(range(len(nuclear_lrmc.index)))
ax.xaxis.set_tick_params(labelsize=14,rotation=90)
ax.yaxis.set_tick_params(labelsize=14)
ax.margins(x=0)
ax.grid(axis = 'y', alpha=0.3)
ax.set_ylabel('€/MWh',fontsize=15)

fig.savefig('../Figures_Power/Key enabling technologies for the green transition Fig 7. Nuclear LRMCs.pdf', bbox_inches='tight')

plt.show()

"""

###########################################################


"""
nuc_cap = pd.read_excel(nuc_cap, sheet_name='Nuclear table', usecols='AD:AS', header = 1)[:41]
nuc_cap['Unnamed: 29'] = nuc_cap['Unnamed: 29'].astype(int)
nuc_cap = nuc_cap.set_index('Unnamed: 29')
nuc_cap.index.name = None
nuc_cap = nuc_cap.loc[2015:]

liste = []
for i in nuc_cap.columns:
    liste.append(nuc_cap[i])


colors = ['navy', 'darkorange', 'darkseagreen', 'firebrick', 'darksalmon',
          'purple', 'orchid', 'steelblue', 'lightblue', 'darkgoldenrod',
          'gainsboro', 'darkgrey','dimgrey', 'yellowgreen','saddlebrown']
fig, ax = plt.subplots(figsize=(8,7))
ax.stackplot(nuc_cap.index, liste, labels = nuc_cap.columns, alpha=0.65, colors = colors)
ax.legend(bbox_to_anchor =(0.5,-0.45),  loc='lower center', ncol=3,fontsize=12.5, frameon=False)


ax.xaxis.set_tick_params(labelsize=14,rotation=90)
ax.yaxis.set_tick_params(labelsize=14)
ax.margins(x=0)
ax.grid(axis = 'y', alpha=0.3)
ax.set_ylabel('MW',fontsize=15)

fig.savefig('../Figures_Power/Key enabling technologies for the green transition Fig 8.  Nuclear Capacity Development.pdf', bbox_inches='tight')

plt.show()
"""

####################################################################

"""
nordic_growth = pd.read_excel(nordic_power,sheet_name='Renewable growth Norway',header=1)
nordic_growth.columns = nordic_growth.columns.astype(str)
nordic_growth = nordic_growth.drop(columns = ['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', ])
nordic_growth.index = nordic_growth.index + 3

nordic_growth = nordic_growth.loc[[74,76,77,78,79,80]]
nordic_growth['Hydro'] = ['Hydro','Onshore wind', 'Bio CHP', 'Utility scale solar', 'Offshore wind,', 'Rooftop solar']
nordic_growth = nordic_growth.set_index('Hydro')
nordic_growth.index.name = None
nordic_growth = nordic_growth.transpose()
print(nordic_growth)

#params = ['Wind onshore', 'Wind offshore', 'Hydro', 'CHP', 'Solar Utility scale', 'Solar Rooftop', 'Nuclear']

y = []
for i in nordic_growth.columns:
    y.append(nordic_growth[i])
    
x = nordic_growth.index

colors = ['navy','lightsteelblue', '#7aa741', '#f4d65c','steelblue','#e3ad12']

fig, ax = plt.subplots(figsize  = (8,7))


ax.stackplot(x,y, colors = colors, alpha = 0.85, labels = nordic_growth.columns)
ax.legend(bbox_to_anchor =(0.5,-0.35), loc='lower center', ncol=3,fontsize=14, frameon=False)
xticks = return_xticks(nordic_growth.index, ['2025','2030','2035','2040','2045','2050'])
ax.set_xticklabels(xticks,rotation = 90,fontsize=14)
ax.yaxis.set_tick_params(labelsize=14)
ax.margins(x=0)
ax.grid(axis = 'y', alpha=0.3)
ax.set_ylabel('TWh',fontsize=15)

fig.savefig('../Figures_Power/The Nordics Fig 2.  Renewable Growth, Norway.pdf', bbox_inches='tight')

plt.show()

"""

###############################################################################

"""
nordic_growth = pd.read_excel(nordic_power,sheet_name='Renewable growth Sweden',header=1)
nordic_growth.columns = nordic_growth.columns.astype(str)
nordic_growth = nordic_growth.drop(columns = ['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', ])
nordic_growth.index = nordic_growth.index + 3

nordic_growth = nordic_growth.loc[[67,69,70,71,72,73]]
nordic_growth['Hydro'] = ['Hydro','Onshore wind', 'Bio CHP', 'Utility scale solar', 'Offshore wind,', 'Rooftop solar']
nordic_growth = nordic_growth.set_index('Hydro')
nordic_growth.index.name = None
nordic_growth = nordic_growth.transpose()
print(nordic_growth)

#params = ['Wind onshore', 'Wind offshore', 'Hydro', 'CHP', 'Solar Utility scale', 'Solar Rooftop', 'Nuclear']

y = []
for i in nordic_growth.columns:
    y.append(nordic_growth[i])
    
x = nordic_growth.index

colors = ['navy','lightsteelblue', '#7aa741', '#f4d65c','steelblue','#e3ad12']

fig, ax = plt.subplots(figsize  = (8,7))


ax.stackplot(x,y, colors = colors, alpha = 0.85, labels = nordic_growth.columns)
ax.legend(bbox_to_anchor =(0.5,-0.35), loc='lower center', ncol=3,fontsize=14, frameon=False)
xticks = return_xticks(nordic_growth.index, ['2025','2030','2035','2040','2045','2050'])
ax.set_xticklabels(xticks,rotation = 90,fontsize=14)
ax.yaxis.set_tick_params(labelsize=14)
ax.margins(x=0)
ax.grid(axis = 'y', alpha=0.3)
ax.set_ylabel('TWh',fontsize=15)

fig.savefig('../Figures_Power/The Nordics Fig 2.  Renewable Growth, Sweden.pdf', bbox_inches='tight')

plt.show()
"""
###################################################################################
"""
nordic_growth = pd.read_excel(nordic_power,sheet_name='Renewable growth Denmark',header=1)
nordic_growth.columns = nordic_growth.columns.astype(str)
nordic_growth = nordic_growth.drop(columns = ['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', ])
nordic_growth.index = nordic_growth.index + 3

nordic_growth = nordic_growth.loc[[50,52,53,54,55,56]]
nordic_growth['Wind Onshore'] = ['Hydro','Onshore wind', 'Bio CHP', 'Utility scale solar', 'Offshore wind,', 'Rooftop solar']
nordic_growth = nordic_growth.set_index('Wind Onshore')
nordic_growth.index.name = None
nordic_growth = nordic_growth.transpose()

print(nordic_growth)

#params = ['Wind onshore', 'Wind offshore', 'Hydro', 'CHP', 'Solar Utility scale', 'Solar Rooftop', 'Nuclear']

y = []
for i in nordic_growth.columns:
    y.append(nordic_growth[i])
    
x = nordic_growth.index

colors = ['navy','lightsteelblue', '#7aa741', '#f4d65c','steelblue','#e3ad12']

fig, ax = plt.subplots(figsize  = (8,7))


ax.stackplot(x,y, colors = colors, alpha = 0.85, labels = nordic_growth.columns)
ax.legend(bbox_to_anchor =(0.5,-0.35), loc='lower center', ncol=3,fontsize=14, frameon=False)
xticks = return_xticks(nordic_growth.index, ['2025','2030','2035','2040','2045','2050'])
ax.set_xticklabels(xticks,rotation = 90,fontsize=14)
ax.yaxis.set_tick_params(labelsize=14)
ax.margins(x=0)
ax.grid(axis = 'y', alpha=0.3)
ax.set_ylabel('TWh',fontsize=15)

fig.savefig('../Figures_Power/The Nordics Fig 2.  Renewable Growth, Denmark.pdf', bbox_inches='tight')

plt.show()
"""
#############################################################################3
"""

nordic_growth = pd.read_excel(nordic_power,sheet_name='Renewable growth Norway',header=1)
nordic_growth.columns = nordic_growth.columns.astype(str)
nordic_growth = nordic_growth.drop(columns = ['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', ])
nordic_growth.index = nordic_growth.index + 3

norway = [74,76,77,78,79,80]
sweden = [67,69,70,71,72,73]

nordic_growth = nordic_growth.loc[norway]
print(nordic_growth)
nordic_growth['Hydro'] = ['Hydro','Onshore wind', 'Bio CHP', 'Utility scale solar', 'Offshore wind', 'Rooftop solar']
nordic_growth = nordic_growth.set_index('Hydro')
nordic_growth.index.name = None
nordic_growth = nordic_growth.transpose().div(1000)
nordic_growth = nordic_growth[['Hydro', 'Onshore wind', 'Offshore wind', 'Bio CHP', 'Utility scale solar', 'Rooftop solar']].copy()


print(nordic_growth)

#params = ['Wind onshore', 'Wind offshore', 'Hydro', 'CHP', 'Solar Utility scale', 'Solar Rooftop', 'Nuclear']

y = []
for i in nordic_growth.columns:
    y.append(nordic_growth[i])
    
x = nordic_growth.index

colors = ['navy','lightsteelblue', 'steelblue', '#7aa741','#f4d65c','#e3ad12']

fig, ax = plt.subplots(figsize  = (8,5))


ax.stackplot(x,y, colors = colors, alpha = 0.85, labels = nordic_growth.columns)
ax.legend(bbox_to_anchor =(0.5,-0.35), loc='lower center', ncol=3,fontsize=14, frameon=False)
xticks = return_xticks(nordic_growth.index, ['2025','2030','2035','2040','2045','2050'])
ax.set_xticklabels(xticks,rotation = 90,fontsize=14)
ax.yaxis.set_tick_params(labelsize=14)
ax.margins(x=0)
ax.grid(axis = 'y', alpha=0.3)
ax.set_ylabel('TWh',fontsize=15)
ax.set_ylim(0,25)
fig.savefig('../Figures_Power/The Nordics Fig 2.  Renewable Growth, Norway.pdf', bbox_inches='tight')

plt.show()
"""
######################################################################################################
############################### RENEWABLE GROWTH DENMARK #############################################
"""

nordic_growth = pd.read_excel(nordic_power,sheet_name='Renewable growth Denmark',header=1)
nordic_growth.columns = nordic_growth.columns.astype(str)
nordic_growth = nordic_growth.drop(columns = ['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', ])
nordic_growth.index = nordic_growth.index + 3



denmark = [50,52,53,54,55,56]



nordic_growth = nordic_growth.loc[denmark]

nordic_growth['Wind Onshore'] = ['Hydro','Onshore wind', 'Bio CHP', 'Utility scale solar', 'Offshore wind', 'Rooftop solar']
nordic_growth = nordic_growth.set_index('Wind Onshore')
nordic_growth.index.name = None
nordic_growth = nordic_growth.transpose().div(1000)
nordic_growth = nordic_growth[['Hydro', 'Onshore wind', 'Offshore wind', 'Bio CHP', 'Utility scale solar', 'Rooftop solar']].copy()
print(nordic_growth)


y = []
for i in nordic_growth.columns:
    y.append(nordic_growth[i])
    
x = nordic_growth.index

colors = ['navy','lightsteelblue', 'steelblue', '#7aa741','#f4d65c','#e3ad12']

fig, ax = plt.subplots(figsize  = (8,5))


ax.stackplot(x,y, colors = colors, alpha = 0.85, labels = nordic_growth.columns)
ax.legend(bbox_to_anchor =(0.5,-0.35), loc='lower center', ncol=3,fontsize=14, frameon=False)
xticks = return_xticks(nordic_growth.index, ['2025','2030','2035','2040','2045','2050'])
ax.set_xticklabels(xticks,rotation = 90,fontsize=14)
ax.yaxis.set_tick_params(labelsize=14)
ax.margins(x=0)
ax.grid(axis = 'y', alpha=0.3)
ax.set_ylabel('TWh',fontsize=15)
ax.set_ylim(0,25)
fig.savefig('../Figures_Power/The Nordics Fig 4.  Renewable Growth, Denmark.pdf', bbox_inches='tight')

plt.show()

######################################################################################################
############################### RENEWABLE GROWTH FINLAND #############################################

nordic_growth = pd.read_excel(nordic_power,sheet_name='Renewable growth Finland',header=0)
nordic_growth.columns = nordic_growth.columns.astype(str)
nordic_growth = nordic_growth.drop(columns = ['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', ])
nordic_growth.index = nordic_growth.index + 2


finland = [30,31,32,33,34,35]



nordic_growth = nordic_growth.loc[finland]

nordic_growth['Renewable growth TWh'] = ['Hydro','Onshore wind', 'Bio CHP', 'Utility scale solar', 'Offshore wind', 'Rooftop solar']
nordic_growth = nordic_growth.set_index('Renewable growth TWh')
nordic_growth.index.name = None
nordic_growth = nordic_growth.transpose().div(1000)
nordic_growth = nordic_growth[['Hydro', 'Onshore wind', 'Offshore wind', 'Bio CHP', 'Utility scale solar', 'Rooftop solar']].copy()

y = []
for i in nordic_growth.columns:
    y.append(nordic_growth[i])
    
x = nordic_growth.index

colors = ['navy','lightsteelblue', 'steelblue', '#7aa741','#f4d65c','#e3ad12']

fig, ax = plt.subplots(figsize  = (8,5))


ax.stackplot(x,y, colors = colors, alpha = 0.85, labels = nordic_growth.columns)
ax.legend(bbox_to_anchor =(0.5,-0.35), loc='lower center', ncol=3,fontsize=14, frameon=False)
xticks = return_xticks(nordic_growth.index, ['2025','2030','2035','2040','2045','2050'])
ax.set_xticklabels(xticks,rotation = 90,fontsize=14)
ax.yaxis.set_tick_params(labelsize=14)
ax.margins(x=0)
ax.grid(axis = 'y', alpha=0.3)
ax.set_ylabel('TWh',fontsize=15)
ax.set_ylim(0,30)
fig.savefig('../Figures_Power/The Nordics Fig 5.  Renewable Growth, Finland.pdf', bbox_inches='tight')

plt.show()

"""
####################################################################################
"""
poland = pd.read_excel(nordic_power,sheet_name = 'Base case')
poland.index = poland.index + 2
poland = poland.loc[26]
poland = poland.loc[2025:]

rest = pd.read_excel(nordic_power, sheet_name = 'Eastern Europe')
rest.index = rest.index + 2

estonia = rest.loc[10].loc[2025:]
latvia = rest.loc[24].loc[2025:]
lithuania = rest.loc[38].loc[2025:]

fig, ax = plt.subplots(figsize=(8,7))
estonia.plot(ax = ax, linewidth = 2, color = 'firebrick', label = 'Estonia')
poland.plot(ax = ax, linewidth = 2, color = 'darkorange', label = 'Poland')
latvia.plot(ax = ax, linewidth = 2, color = 'gray', label = 'Latvia')
lithuania.plot(ax = ax, linewidth = 2, color = 'navy', label = 'Lithuania')


ax.legend(bbox_to_anchor =(0.5,-0.25),  loc='lower center', ncol=4,fontsize=12.5, frameon=False)

ax.set_xticks(range(len(estonia.index)))
ax.xaxis.set_tick_params(labelsize=14,rotation=90)
ax.yaxis.set_tick_params(labelsize=14)
ax.margins(x=0)
ax.grid(axis = 'y', alpha=0.3)
ax.set_ylabel('€/MWh',fontsize=15)
ax.set_ylim(40,120)

#plt.show()
fig.savefig('../Figures_Power/Baltic and Eastern Europe fig 1. Eastern European price projections.pdf', bbox_inches='tight')

print(estonia)

"""