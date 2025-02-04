
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
#sns.set()

plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
#plt.rcParams['axes.spines.bottom'] = False


def return_xticks(all_years, specific_years):
    new_xticks = []
    for i in all_years:
        if i in specific_years:
            new_xticks.append(i)
        else:
            new_xticks.append('')
    return new_xticks
    


input_cost = r'M:\Prosjekter\Long Term Nordic\2024 Fall\Input costs Nordic Fall 2024.xlsx'
results = r'M:\LongTerm\nena-lrmcmodel\results\Fall 2024\results.xlsx'
nordic_growth_pth = r'M:\Prosjekter\Green Certificates\2015 October\Forward exchange rates_Longterms_Fall_2015.xlsx'

nordic_power = r'M:\Prosjekter\Long Term Nordic\2024 Fall\Nordic Power Market Outlook - 2024-2050 (Fall 24) LINKS.xlsx'

hourly_prices = r'M:\Prosjekter\Long Term Nordic\2024 Fall\Hourly prices Base Fall 2024.xlsx'
average = r'M:\Prosjekter\Green Certificates\2024 October\Katarina\2001-2020 avg prices.xlsx'


base = pd.read_excel(input_cost, sheet_name = 'Base Scenario')
base.columns = base.columns.astype(str)
lrmc_df = base.drop(columns=['Unnamed: 1', 'Unnamed: 2', '2023', '2024', 'Unnamed: 31'])



val_results = pd.read_excel(results, header=2)
val_results.columns = val_results.columns.astype(str)
val_results = val_results.rename(columns={'Unnamed: 0': 'params'})
val_results = val_results.drop(columns=['Unnamed: 1', 'Unnamed: 2', '2024'])



nordic_growth = pd.read_excel(nordic_growth_pth,sheet_name='Nordics',header=1)
nordic_growth = nordic_growth[37:48]
nordic_growth.columns = nordic_growth.columns.astype(str)
nordic_growth = nordic_growth.rename(columns={'Investments (TWh)': 'params'})
nordic_growth = nordic_growth.drop(columns=['2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024'])


german_pp = r'M:\EUROPA\Germany\German power plants and installed capacity.xlsx'

"""
######################################################################
params = ['Wind onshore', 'Wind offshore', 'Hydro', 'CHP', 'Solar Utility scale', 'Solar Rooftop', 'Nuclear']

indices = []
for i in params:
    index = nordic_growth.index[nordic_growth['params'] == i].values[0]
    indices.append(index)


nordic = nordic_growth.loc[indices]
nordic = nordic.set_index('params')
nordic.index.name = None
nordic = nordic.transpose()


fig1, ax = plt.subplots(figsize = (14,8))


x = nordic.index
y = [nordic['Hydro'], nordic['CHP'], nordic['Wind onshore'], nordic['Wind offshore'], nordic['Solar Utility scale'],\
    nordic['Solar Rooftop']]


color_map = ['darkblue','green','lightsteelblue', 'steelblue', 'yellow', 'orange']
label = ['Hydro', 'Bio CHP', 'Onshore wind', 'Offshore wind', 'Utility scale solar', 'Rooftop solar']


ax.stackplot(x,y, colors = color_map, alpha = 0.6, labels = label)


ax.legend(bbox_to_anchor =(0.5,-0.25), loc='lower center', ncol=3,fontsize=17, frameon=False)
xticks = return_xticks(nordic.index, ['2025','2030','2035','2040','2045','2050'])
ax.set_xticklabels(xticks,rotation = 90,fontsize=17)
ax.yaxis.set_tick_params(labelsize=17)
ax.margins(x=0)
ax.grid(axis = 'y', alpha=0.3)
ax.set_ylabel('TWh',fontsize=17)



fig1.savefig('../Figurer_presentasjon/Summarizing chapter Fig 1. Nordic Renewable Growth.png', bbox_inches='tight')
"""
#######################################################################
"""
nordics_lrmc_vars = ['Onshore wind Nordics', 'Offshore wind Nordic West (bottom fixed)', 'Offshore wind Nordic West (floating)', 'PV (Roof top) Nordics', 'PV (Utility scale) Nordics', 'Bio CHP', r'CCGT (52.6 % Eff.)', r'Coal (38.8 % Eff.)']


indices = []
for i in nordics_lrmc_vars:
    index = base.index[base['params'] == i].values[0]
    indices.append(index)


nordics_lrmc_df = lrmc_df.loc[indices]
nordics_lrmc_df = nordics_lrmc_df.set_index('params')
nordics_lrmc_df.index.name = None
nordics_lrmc_df = nordics_lrmc_df.transpose()


fig2, ax1 = plt.subplots(figsize = (14,8))


nordics_lrmc_df['Onshore wind Nordics'].plot(ax = ax1, 
                                             color = 'skyblue', 
                                             linewidth= 4,
                                             label = 'LRMC Onshore wind Nordics (3410 FLH,WACC 6.2%)')
nordics_lrmc_df['Offshore wind Nordic West (bottom fixed)'].plot(ax = ax1, dashes=[6, 2],
                                                                 linewidth = 4,
                                                                 color = 'steelblue',
                                                                 label = 'LRMC Offshore wind Nordic West (bottom fixed) (4509 FLH, WACC 7.5%)')
nordics_lrmc_df['Offshore wind Nordic West (floating)'].plot(ax=ax1,
                                                             color = 'steelblue',
                                                             linewidth = 4,
                                                             label = 'LRMC Offshore wind Nordic West (floating) (4509 FLH, WACC 7.5%)')
nordics_lrmc_df['PV (Roof top) Nordics'].plot(ax = ax1,
                                              color = 'orange',
                                              linewidth = 4,
                                              label = 'LRMC PV (Roof top) Nordics (959 FLH, WACC 5.7%)')
nordics_lrmc_df['PV (Utility scale) Nordics'].plot(ax = ax1, dashes = [6,2],
                                                   color = 'orange',
                                                   linewidth = 4,
                                                   label = 'LRMC PV (Utility scale) Nordics (1020 FLH, WACC 6.2%)')
nordics_lrmc_df['CCGT (52.6 % Eff.)'].plot(ax = ax1,
                                color = 'saddlebrown',
                                linewidth = 4,
                                label = 'SRMC CCGT')
nordics_lrmc_df['Coal (38.8 % Eff.)'].plot(ax = ax1,
                                color = 'black',
                                linewidth = 4,
                                label = 'SRMC Coal')


fig2.legend(bbox_to_anchor =(0.5,-0.3),  loc='lower center', ncol=1,fontsize=17, frameon=False)

ax1.set_xticks(range(len(nordics_lrmc_df.index)))
ax1.xaxis.set_tick_params(labelsize=17,rotation=90)
ax1.yaxis.set_tick_params(labelsize=17)
ax1.margins(x=0)
ax1.grid(axis = 'y', alpha=0.3)
ax1.set_ylabel('€/MWh',fontsize=17)

fig2.savefig('../Figurer_presentasjon/fossil energy priced out.png', bbox_inches='tight')
"""

"""
#####################################################################

# LRMCs Onshore Wind Power



wind_lrmc = lrmc_df.iloc[[61,62,70]]

wind_lrmc = wind_lrmc.set_index('params')
wind_lrmc.index.name = None
wind_lrmc = wind_lrmc.transpose()

wind_lrmc = wind_lrmc.rename(columns={'Onshore wind Norway': 'Onshore wind Norway (3511 FLH, WACC 6,2%)',\
                                      'Onshore wind Nordics': 'Onshore wind Nordics (3410 FLH, WACC 6,2 %)',\
                                      'Onshore wind Conti': ' Onshore wind Conti (2909 FLH, WACC 6,2%)'})


fig, ax = plt.subplots(figsize = (14,8))
colors = ['darkseagreen', 'darkgoldenrod','darkolivegreen']
wind_lrmc.plot(ax = ax, color = colors, linewidth = 4)


ax.legend(bbox_to_anchor =(0.5,-0.32),  loc='lower center', ncol=1,fontsize=17, frameon=False)

ax.set_xticks(range(len(wind_lrmc.index)))
ax.xaxis.set_tick_params(labelsize=17,rotation=90)
ax.yaxis.set_tick_params(labelsize=17)
ax.margins(x=0)
ax.grid(axis = 'y', alpha=0.3)
ax.set_ylabel('€/MWh',fontsize=17)

fig.savefig('../Figurer_presentasjon/OnshoreWind LRMC Onshore Wind Power.png', bbox_inches='tight')

#plt.show()
#############################################################################
## LRMC Nuclear

nuclear_lrmc = lrmc_df.iloc[[85,86]]
nuclear_lrmc = nuclear_lrmc.set_index('params')
nuclear_lrmc.index.name = None
nuclear_lrmc = nuclear_lrmc.transpose()

nuclear_lrmc = nuclear_lrmc.rename(columns={'New Nuclear': 'New Nuclear (8059 FLH, WACC 8,5%)',\
                                      'Nuclear LTO': 'Nuclear LTO (7446 FLH, WACC 7,5 %)'})

new_nuclear = nuclear_lrmc[['New Nuclear (8059 FLH, WACC 8,5%)']].copy()
nuclear_lto = nuclear_lrmc[['Nuclear LTO (7446 FLH, WACC 7,5 %)']].copy()

fig, ax = plt.subplots(figsize=(14,8))

new_nuclear.plot(ax = ax, color = 'darkslateblue', linewidth = 4)
nuclear_lto.plot(ax = ax, color = 'darkslateblue', linewidth = 4, dashes = [6, 2])


ax.legend(bbox_to_anchor =(0.5,-0.25),  loc='lower center', ncol=1,fontsize=17, frameon=False)

ax.set_xticks(range(len(nuclear_lrmc.index)))
ax.xaxis.set_tick_params(labelsize=17,rotation=90)
ax.yaxis.set_tick_params(labelsize=17)
ax.margins(x=0)
ax.grid(axis = 'y', alpha=0.3)
ax.set_ylabel('€/MWh',fontsize=17)

fig.savefig('../Figurer_presentasjon/NuclearPower LRMC.png', bbox_inches='tight')


################################################################

hydrogen_vars = ['AEL Flat (50 €/MWh input)', 'PEMEL Flexible (15 €/MWh input)', 'SOEC Flat (50 €/MWh input)', 'SRMC Grey H2 €/MWh']


indices = []
for i in hydrogen_vars:
    index = base.index[base['params'] == i].values[0]
    indices.append(index)


hydrogen_df = lrmc_df.loc[indices]
hydrogen_df = hydrogen_df.set_index('params')

index_srmc_blue_base = val_results.index[val_results['params'] == 'SMR_CCS SRMC'].values[0]
srmc_blue = val_results.loc[[index_srmc_blue_base]]
srmc_blue = srmc_blue.set_index('params')

hydrogen_df = pd.concat([hydrogen_df, srmc_blue])
hydrogen_df.index.name = None
hydrogen_df = hydrogen_df.transpose()
hydrogen_df = hydrogen_df.rename(columns={'SMR_CCS SRMC': 'SRMC Blue H2 €/MWh'})




print(hydrogen_df.columns)
hydrogen_df.columns = ['AEL Flat (50 €/MWh input) (8322 FLH, WACC 7.5%)', 
                        'PEMEL Flexible (15 €/MWh input) (3000 FLH, WACC 7.5%)',
                        'SOEC Flat (50 €/MWh input) (8322 FLH, WACC 7.5%)', 
                        'SRMC Grey H2 €/MWh (8322 FLH, WACC 8,7%)',
                        'SRMC Blue H2 €/MWh (8322 FLH, WACC 8%)']
hygen_df = hydrogen_df[['AEL Flat (50 €/MWh input) (8322 FLH, WACC 7.5%)', 
                        'SOEC Flat (50 €/MWh input) (8322 FLH, WACC 7.5%)',
                        'SRMC Grey H2 €/MWh (8322 FLH, WACC 8,7%)',
                        'SRMC Blue H2 €/MWh (8322 FLH, WACC 8%)']]
colors = {}
map = ['#9cc674', 'darkgreen', 'grey', '#0264b1']
for i in range(len(hygen_df.columns)):
    colors[hygen_df.columns[i]] = map[i]
print(colors) 

fig5,ax2 = plt.subplots(figsize = (8,7))

hygen_df.plot(ax = ax2, color = map, linewidth = 2)
hydrogen_df['PEMEL Flexible (15 €/MWh input) (3000 FLH, WACC 7.5%)'].plot(ax=ax2, color = '#9cc674',dashes=[6, 2], linewidth =2)
ax2.legend(bbox_to_anchor =(0.5,-0.45), loc='lower center', ncol=1,fontsize=13, frameon=False)
ax2.set_xticks(range(len(hygen_df.index)))
ax2.xaxis.set_tick_params(labelsize=14,rotation=90)
ax2.yaxis.set_tick_params(labelsize=14)
ax2.margins(x=0)
ax2.grid(axis = 'y', alpha=0.3)
ax2.set_ylabel('€/MWh',fontsize=15)


fig5.savefig('../Figures_Renewables/Summarizing chapter Fig 5. LRMCs green vs grey and blue.pdf', bbox_inches='tight')
fig5.savefig('../Figures_Renewables/Hydrogen Fig 2. LRMCs green vs grey and blue.pdf', bbox_inches='tight')



plt.show()
"""

#### Figure 5, LRMCs grey vs grey and blue ####

hydrogen_vars = ['AEL Flat (50 €/MWh input)', 'PEMEL Flexible (15 €/MWh input)', 'SOEC Flat (50 €/MWh input)', 'SRMC Grey H2','LRMC Blue H2 €/MWh']

print(base)
indices = []
for i in hydrogen_vars:
    index = base.index[base['params'] == i].values[0]
    indices.append(index)



hydrogen_df = lrmc_df.loc[indices]
hydrogen_df = hydrogen_df.set_index('params')

#print(hydrogen_df)
#index_srmc_blue_base = val_results.index[val_results['params'] == 'SMR_CCS SRMC'].values[0]
#srmc_blue = val_results.loc[[index_srmc_blue_base]]
#srmc_blue = srmc_blue.set_index('params')

hydrogen_df = hydrogen_df
hydrogen_df.index.name = None
hydrogen_df = hydrogen_df.transpose()
#hydrogen_df = hydrogen_df.rename(columns={'LRMC Blue H2 €/MWh': 'SRMC Blue H2 €/MWh'})




print(hydrogen_df.columns)
hydrogen_df.columns = ['AEL Flat (50 €/MWh input) (8322 FLH, WACC 7.5%)', 
                        'PEMEL Flexible (15 €/MWh input) (3000 FLH, WACC 7.5%)',
                        'SOEC Flat (50 €/MWh input) (8322 FLH, WACC 7.5%)', 
                        'SRMC Grey H2 €/MWh (8322 FLH, WACC 8,7%)',
                        'LRMC Blue H2 €/MWh (8322 FLH, WACC 8%)']


hygen_df = hydrogen_df[['AEL Flat (50 €/MWh input) (8322 FLH, WACC 7.5%)', 
                        'SOEC Flat (50 €/MWh input) (8322 FLH, WACC 7.5%)',
                        'SRMC Grey H2 €/MWh (8322 FLH, WACC 8,7%)',
                        'LRMC Blue H2 €/MWh (8322 FLH, WACC 8%)']]
colors = {}
map = ['#9cc674', 'darkgreen', 'grey', '#0264b1']

for i in range(len(hygen_df.columns)):
    colors[hygen_df.columns[i]] = map[i]
print(colors) 

fig5,ax2 = plt.subplots(figsize = (14,8))

hygen_df.plot(ax = ax2, color = map, linewidth = 4)
hydrogen_df['PEMEL Flexible (15 €/MWh input) (3000 FLH, WACC 7.5%)'].plot(ax=ax2, color = '#9cc674',dashes=[6, 2], linewidth =4)
ax2.legend(bbox_to_anchor =(0.5,-0.45), loc='lower center', ncol=1,fontsize=17, frameon=False)
ax2.set_xticks(range(len(hygen_df.index)))
ax2.xaxis.set_tick_params(labelsize=17,rotation=90)
ax2.yaxis.set_tick_params(labelsize=17)
ax2.margins(x=0)
ax2.grid(axis = 'y', alpha=0.3)
ax2.set_ylabel('€/MWh',fontsize=17)


fig5.savefig('../Figurer_presentasjon/LRMCs green vs grey and blue.png', bbox_inches='tight')

#### Figure 5, Hydrogen Growth Capacity And Electricity ####

"""
hdrg = pd.read_excel(nordic_growth_pth, sheet_name = "Hdrg Nordics", header = 4)


indices = [42,45,26,29]
hdrg = hdrg.iloc[indices]
hdrg.columns = hdrg.columns.astype(str)
hdrg = hdrg.drop(columns=['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', '2022', '2023', '2024'])
hdrg['vars'] = ['Installed Capacity (Flat) GW', 'Installed Capacity (Flex) GW', 'Electricity Demand (Flat) TWh', 'Electricty Demand (Flex) TWh']
hdrg = hdrg.set_index('vars').drop(columns='Hydrogen flat TWh el')
hdrg.index.name = None

remove = ['La stå']
for i in hdrg.columns:
    if 'Unnamed' in i:
        remove.append(i)

hdrg = hdrg.drop(columns=remove)
hdrg = hdrg.transpose()
hdrg['Installed Capacity (Flat) GW'] = hdrg['Installed Capacity (Flat) GW']/1000
hdrg['Installed Capacity (Flex) GW'] = hdrg['Installed Capacity (Flex) GW']/1000
hdrg['Total Electricity Demand TWh'] = hdrg['Electricity Demand (Flat) TWh'] + hdrg['Electricty Demand (Flex) TWh']


fig, ax1 = plt.subplots(figsize = (14,8))





l1 = ax1.stackplot(hdrg.index, [hdrg['Installed Capacity (Flat) GW'], hdrg['Installed Capacity (Flex) GW']], alpha = 0.33, colors=['#183200', '#adc893'])
ax1.yaxis.set_tick_params(labelsize=17)
ax1.margins(x=0)
xticks = return_xticks(hdrg.index, ['2025','2030','2035','2040','2045','2050'])
ax1.set_xticklabels(xticks,rotation = 90,fontsize=17)
ax1.grid(axis = 'y', alpha=0.3)
ax1.set_ylabel('GW',rotation=0,fontsize=17)
ax1.yaxis.set_label_coords(-0.05, 1.0125)

ax2 = ax1.twinx()

l2 = ax2.plot(hdrg.index,  hdrg['Electricty Demand (Flex) TWh'], color = 'gold', linewidth = 4)
l3 = ax2.plot(hdrg.index,  hdrg['Electricity Demand (Flat) TWh'], color = 'saddlebrown', linewidth = 4)
l4 = ax2.plot(hdrg.index,  hdrg['Total Electricity Demand TWh'], color = 'grey', linewidth = 4)
ax2.set_ylim(bottom = 0)
ax2.set_ylabel('TWh',rotation=0, fontsize=17)
ax2.yaxis.set_label_coords(1.05, 1.05)

ax2.yaxis.set_tick_params(labelsize=14)

fig.legend([l1,l2,l3,l4], labels=['Installed Capacity (Flat) GW', 
                                  'Installed Capacity (Flex) GW',
                                  'Electricity Demand (Flex) TWh', 
                                  'Electricity Demand (Flat) TWh',
                                  'Total Electricity Demand TWh'],
                                  bbox_to_anchor =(0.5,-0.15), loc='lower center', ncol=2,fontsize=17, frameon=False) 
ax2.tick_params(left = False, right=False)
ax1.tick_params(left = False, right=False)
fig.savefig('../Figurer_presentasjon/ Hydrogen Growth Capacity and Electricity.png', bbox_inches='tight')
"""

###########################################################


### LRMCs of flexible generation technologies
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

fig4,ax2 = plt.subplots(figsize = (14,8))

flex_lrmc.plot(ax = ax2, linewidth=4)
ax2.legend(bbox_to_anchor =(0.5,-0.45), loc='lower center', ncol=1,fontsize=17, frameon=False)
ax2.set_xticks(range(len(flex_lrmc.index)))
ax2.xaxis.set_tick_params(labelsize=14,rotation=90)
ax2.yaxis.set_tick_params(labelsize=14)
ax2.margins(x=0)
ax2.grid(axis = 'y', alpha=0.3)
ax2.set_ylabel('€/MWh',fontsize=17)


fig4.savefig('../Figurer_presentasjon/Flexible Production.png', bbox_inches='tight')
"""
#################################################################################

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

fig, ax = plt.subplots(figsize=(14,8))
labels = ['Hydrogen', 'Electric vehicles', 'Data Storage', 'Industry', 'Losses', 'Petroleum Sector']
ax.stackplot(cons.index, [cons['Hydrogen'], cons['Electric vehicles'],
                           cons['Data Storage'], cons['Industry'],
                           cons['Losses incl. Pumped Storage'], cons['Petroleum Sector']], colors = colors, labels = labels)


ax.legend(bbox_to_anchor =(0.5,-0.35), loc='lower center', ncol=2,fontsize=17, frameon=False)
xticks = return_xticks(cons.index, ['2025','2030','2035','2040','2045','2050'])
ax.set_xticklabels(xticks,rotation = 90,fontsize=17)
ax.yaxis.set_tick_params(labelsize=17)
ax.margins(x=0)
ax.grid(axis = 'y', alpha=0.3)
ax.set_ylabel('TWh',fontsize=17)

fig.savefig('../Figurer_presentasjon/Electricity consumption growth 2025-2050.png', bbox_inches='tight')

"""

##############################################################################
"""
net_exchange = pd.read_excel(nordic_power, sheet_name='Base case')
net_exchange = net_exchange.drop(columns=['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5'])
net_exchange.index = net_exchange.index + 2

net_exchange = net_exchange.loc[[34]]
net_exchange = net_exchange.set_index('€/MWh')
net_exchange.index.name = None

net_exchange = net_exchange.transpose()
net_exchange = net_exchange[1:]




fig, ax= plt.subplots(figsize=(14,8))
net_exchange.plot(ax = ax, color = "darkorange", linewidth = 4)
ax.legend(bbox_to_anchor =(0.5,-0.25),  loc='lower center', ncol=1,fontsize=17, frameon=False)

ax.set_xticks(range(len(net_exchange.index)))
ax.xaxis.set_tick_params(labelsize=17,rotation=90)  
ax.yaxis.set_tick_params(labelsize=17)
ax.margins(x=0)
ax.grid(axis = 'y', alpha=0.3)
ax.set_ylabel('TWh',fontsize=17)
ax.set_ylim(-100,-20)

fig.savefig('../Figurer_presentasjon/Nordic Power Base Scenario Net Exchange.png', bbox_inches='tight')
"""

##################################################################################
"""
summary = pd.read_excel(german_pp, sheet_name = 'LT misc', header = 1)
summary.columns = summary.columns.astype(str)
summary = summary[['€/MWh', '2025', '2030', '2035', '2040', '2050']].copy()
summary.index = summary.index + 3




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

    fig, ax = plt.subplots(figsize = (14,8))

    table.plot(ax = ax, linewidth = 4)
    n = 2
    if table_name == 'SRMCs_and_EUAs':
        n = 3
    ax.legend(bbox_to_anchor =(0.5,-0.3),  loc='lower center', ncol=n,fontsize=17, frameon=False)

    ax.set_xticks(range(len(table.index)))
    ax.xaxis.set_tick_params(labelsize=17,rotation=90)
    ax.yaxis.set_tick_params(labelsize=17)
    ax.margins(x=0)
    ax.grid(axis = 'y', alpha=0.3)
    ax.set_ylabel('{}'.format(y),fontsize=15)

    if table_name == 'net_exchange':
        ax.set_ylim(-60,20)

    name = '{}'.format(table_name.replace('_', ' ')).capitalize()
    fig.savefig('../Figurer_presentasjon/Executive Summary fig {} {}.png'.format(count, name), bbox_inches='tight')

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


"""

#################################################################

"""
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


fig, ax= plt.subplots(figsize=(14,8))
percentiles.plot(ax = ax, linewidth = 4)
ax.legend(bbox_to_anchor =(0.5,-0.25),  loc='lower center', ncol=4,fontsize=17, frameon=False)

#ax.set_xticks(range(len(percentiles.index)))
ax.xaxis.set_tick_params(labelsize=17,rotation=0)  
ax.yaxis.set_tick_params(labelsize=17)
ax.margins(x=0)
ax.grid(axis = 'y', alpha=0.3)
ax.set_ylabel('€/MWh',fontsize=17)

fig.savefig('../Figurer_presentasjon/Nordic Power Base Scenario System price distribution.png', bbox_inches='tight')

"""

###########################################################################



"""
gernor = pd.read_excel(nordic_power, sheet_name = 'Base case', header = 0)

vars = ['Nordic', 'German base', 'Coal', 'CCGT']
indices = []
for i in vars:
    index = gernor.index[gernor['€/MWh'] == i].values[0]
    indices.append(index)


gernor.columns = gernor.columns.astype(str)
gernor = gernor.loc[indices]
gernor = gernor.set_index('€/MWh').drop(columns = ['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', '2024'])
gernor.index.name = None
gernor = gernor.transpose()
gernor.columns = ['Nordic System Price', 'German Base', 'SRMC Coal', 'SRMC CCGT']

fig, ax = plt.subplots(figsize=(14,8))
gernor.plot(ax = ax, linewidth = 4)

ax.legend(bbox_to_anchor =(0.5,-0.25),  loc='lower center', ncol=4,fontsize=17, frameon=False)

ax.set_xticks(range(len(gernor.index)))
ax.xaxis.set_tick_params(labelsize=17,rotation=90)  
ax.yaxis.set_tick_params(labelsize=17)
ax.margins(x=0)
ax.grid(axis = 'y', alpha=0.3)
ax.set_ylabel('€/MWh',fontsize=17)

fig.savefig('../Figurer_presentasjon/Nordic_German_Coal_CCGT.png', bbox_inches='tight')


plt.show()
print(gernor)
"""

###########################################################

"""
basecase = pd.read_excel(nordic_power, sheet_name = 'Base case', header = 0)

vars = ['Nordic']
indices = []
for i in vars:
    index = basecase.index[basecase['€/MWh'] == i].values[0]
    indices.append(index)


basecase.columns = basecase.columns.astype(str)
basecase = basecase.loc[indices]
basecase = basecase.set_index('€/MWh').drop(columns = ['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', '2024'])
basecase.index.name = None
basecase = basecase.transpose()
basecase.columns = ['Nordic System Price',]


stromgeo = pd.read_excel(nordic_power, sheet_name = 'StormGeo Best Guess scenario', header = 0)

vars = ['Nordic']
indices = []
for i in vars:
    index = stromgeo.index[stromgeo['€/MWh'] == i].values[0]
    indices.append(index)


stromgeo.columns = stromgeo.columns.astype(str)
stromgeo = stromgeo.loc[indices]
stromgeo = stromgeo.set_index('€/MWh').drop(columns = ['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', '2024'])
stromgeo.index.name = None
stromgeo = stromgeo.transpose()
stromgeo.columns = ['Nordic System Price',]


stormgeo_base = pd.concat([basecase, stromgeo], axis=1)
stormgeo_base.columns = ['Base', 'StormGeo Scenario']

fig, ax = plt.subplots(figsize=(14,8))
stormgeo_base.plot(ax = ax, linewidth = 4, color = ['steelblue', 'lightblue'])

ax.legend(bbox_to_anchor =(0.5,-0.25),  loc='lower center', ncol=2,fontsize=17, frameon=False)

ax.set_xticks(range(len(stormgeo_base.index)))
ax.xaxis.set_tick_params(labelsize=17,rotation=90)  
ax.yaxis.set_tick_params(labelsize=17)
ax.margins(x=0)
ax.grid(axis = 'y', alpha=0.3)
ax.set_ylabel('€/MWh',fontsize=17)

fig.savefig('../Figurer_presentasjon/stormgeo_base.png', bbox_inches='tight')


plt.show()
"""

####################################################################
"""
basecase = pd.read_excel(nordic_power, sheet_name = 'Base case', header = 0)

vars = ['Nordic']
indices = []
for i in vars:
    index = basecase.index[basecase['€/MWh'] == i].values[0]
    indices.append(index)


basecase.columns = basecase.columns.astype(str)
basecase = basecase.loc[indices]
basecase = basecase.set_index('€/MWh').drop(columns = ['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', '2024'])
basecase.index.name = None
basecase = basecase.transpose()
basecase.columns = ['Nordic System Price',]





stromgeo = pd.read_excel(nordic_power, sheet_name = 'StormGeo Best Guess scenario', header = 0)

vars = ['Nordic']
indices = []
for i in vars:
    index = stromgeo.index[stromgeo['€/MWh'] == i].values[0]
    indices.append(index)


stromgeo.columns = stromgeo.columns.astype(str)
stromgeo = stromgeo.loc[indices]
stromgeo = stromgeo.set_index('€/MWh').drop(columns = ['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', '2024'])
stromgeo.index.name = None
stromgeo = stromgeo.transpose()
stromgeo.columns = ['Nordic System Price',]







high = pd.read_excel(nordic_power, sheet_name = 'High scenario', header = 0)

vars = ['Nordic']
indices = []
for i in vars:
    index = high.index[high['€/MWh'] == i].values[0]
    indices.append(index)


high.columns = high.columns.astype(str)
high = high.loc[indices]
high = high.set_index('€/MWh').drop(columns = ['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', '2024'])
high.index.name = None
high = high.transpose()
high.columns = ['Nordic System Price',]




low = pd.read_excel(nordic_power, sheet_name = 'Low scenario', header = 0)

vars = ['Nordic']
indices = []
for i in vars:
    index = low.index[low['€/MWh'] == i].values[0]
    indices.append(index)


low.columns = low.columns.astype(str)
low = low.loc[indices]
low = low.set_index('€/MWh').drop(columns = ['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', '2024'])
low.index.name = None
low = low.transpose()
low.columns = ['Nordic System Price',]







stormgeo_base = pd.concat([basecase, stromgeo, high, low], axis=1)
stormgeo_base.columns = ['Base', 'StormGeo Scenario', 'High Scenario', 'Low Scenario']

fig, ax = plt.subplots(figsize=(14,8))
stormgeo_base.plot(ax = ax, linewidth = 4, color = ['steelblue', 'lightblue', 'maroon', 'olivedrab'])

ax.legend(bbox_to_anchor =(0.5,-0.25),  loc='lower center', ncol=4,fontsize=17, frameon=False)

ax.set_xticks(range(len(stormgeo_base.index)))
ax.xaxis.set_tick_params(labelsize=17,rotation=90)  
ax.yaxis.set_tick_params(labelsize=17)
ax.margins(x=0)
ax.grid(axis = 'y', alpha=0.3)
ax.set_ylabel('€/MWh',fontsize=17)

fig.savefig('../Figurer_presentasjon/stormgeo_base_high_low.png', bbox_inches='tight')


plt.show()
"""