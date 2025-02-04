
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





#### Figure 1, NORDIC RENEWABLE GROWTH #####

"""
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

#### Figure 2, NORDIC LRMCs #####
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
#nordics_lrmc_df['Bio CHP'].plot(ax = ax1,
#                                color = 'olivedrab',
#                                linewidth = 4,
#                                label = 'Bio CHP (wood chips) with 25 MWel (5500 FLH, WACC 7.5%)')


fig2.legend(bbox_to_anchor =(0.5,-0.3),  loc='lower center', ncol=1,fontsize=17, frameon=False)

ax1.set_xticks(range(len(nordics_lrmc_df.index)))
ax1.xaxis.set_tick_params(labelsize=17,rotation=90)
ax1.yaxis.set_tick_params(labelsize=17)
ax1.margins(x=0)
ax1.grid(axis = 'y', alpha=0.3)
ax1.set_ylabel('€/MWh',fontsize=17)

fig2.savefig('../Figurer_presentasjon/Renewables wind and solar LRMC.png', bbox_inches='tight')
#plt.show()
"""
"""

#### Figure 3, Growth Nordic Lithium Ion ####


nordics = ['Norway','Sweden','DK', 'Finland']
bttry_nordics = []
for i in nordics:
    bttry = pd.read_excel(nordic_growth_pth, sheet_name = "{} bttry".format(i), header = 4)
    
    bttry.columns = bttry.columns.astype(str)
    bttry = bttry.drop(columns={'Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', '2022', '2023', '2024'})

    if i == 'Norway':
        k = 13
    if i == 'Sweden':
        k = 11
    if i =='DK':
        k = 7
    if i == 'Finland':
        k = 3
    
    bttry = bttry.iloc[[k]]
    bttry = bttry.set_index('Li-Ion battery power MW')
    bttry.index.name = None
    bttry_nordics.append(bttry.transpose())

lithium = pd.concat(bttry_nordics,axis=1)
lithium = lithium.div(1000)
x = lithium.index
y = [lithium['NO'], lithium['SE'], lithium['DK'], lithium['FI']]

fig3, ax = plt.subplots(figsize = (8,7))


x = lithium.index
y = [lithium['NO'], lithium['SE'], lithium['DK'], lithium['FI']]

color_map = ['#69bdff', '#0264b1', '#bbd1e3', 'olivedrab']
labels = ['NO', 'SE', 'DK', 'FI']
ax.stackplot(x,y, colors = color_map, alpha = 0.5, labels = labels)

ax.legend(bbox_to_anchor =(0.5,-0.25), loc='lower center', ncol=4,fontsize=14, frameon=False)
xticks = return_xticks(lithium.index, ['2025','2030','2035','2040','2045','2050'])
ax.set_xticklabels(xticks,rotation = 90,fontsize=14)
ax.yaxis.set_tick_params(labelsize=14)
ax.set_ylim(top=10)
ax.margins(x=0)
ax.grid(axis = 'y', alpha=0.3)
ax.set_ylabel('GW',fontsize=15)


fig3.savefig('../Figures_Renewables/Summarizing chapter Fig 4. Growth Nordic Lithium Ion.pdf', bbox_inches='tight')
#plt.show()



#### Figure 5, LRMCs grey vs grey and blue ####

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


#### Figure 5, Hydrogen Growth Capacity And Electricity ####

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


fig, ax1 = plt.subplots(figsize = (8,7))





l1 = ax1.stackplot(hdrg.index, [hdrg['Installed Capacity (Flat) GW'], hdrg['Installed Capacity (Flex) GW']], alpha = 0.33, colors=['#183200', '#adc893'])
ax1.yaxis.set_tick_params(labelsize=14)
ax1.margins(x=0)
xticks = return_xticks(hdrg.index, ['2025','2030','2035','2040','2045','2050'])
ax1.set_xticklabels(xticks,rotation = 90,fontsize=14)
ax1.grid(axis = 'y', alpha=0.3)
ax1.set_ylabel('GW',rotation=0,fontsize=14)
ax1.yaxis.set_label_coords(-0.05, 1.0125)

ax2 = ax1.twinx()

l2 = ax2.plot(hdrg.index,  hdrg['Electricty Demand (Flex) TWh'], color = 'gold', linewidth = 2)
l3 = ax2.plot(hdrg.index,  hdrg['Electricity Demand (Flat) TWh'], color = 'saddlebrown', linewidth = 2)
l4 = ax2.plot(hdrg.index,  hdrg['Total Electricity Demand TWh'], color = 'grey', linewidth = 2)
ax2.set_ylim(bottom = 0)
ax2.set_ylabel('TWh',rotation=0, fontsize=14)
ax2.yaxis.set_label_coords(1.05, 1.05)

ax2.yaxis.set_tick_params(labelsize=14)

fig.legend([l1,l2,l3,l4], labels=['Installed Capacity (Flat) GW', 
                                  'Installed Capacity (Flex) GW',
                                  'Electricity Demand (Flex) TWh', 
                                  'Electricity Demand (Flat) TWh',
                                  'Total Electricity Demand TWh'],
                                  bbox_to_anchor =(0.5,-0.15), loc='lower center', ncol=2,fontsize=14, frameon=False) 
ax2.tick_params(left = False, right=False)
ax1.tick_params(left = False, right=False)
fig.savefig('../Figures_Renewables/Summarizing chapter Fig 6. Hydrogen Growth Capacity and Electricity.pdf', bbox_inches='tight')
fig.savefig('../Figures_Renewables/Hydrogen Fig 8. Hydrogen Growth Capacity and Electricity.pdf', bbox_inches='tight')

#print(hydrogen_df)
plt.show()

##############################################################################################################################################################################################



#### Figure 6, EU ETS 1-Price Trajectory ####

EUA = lrmc_df.iloc[[2]]
EUA = EUA.set_index('params')
EUA.index.name = None
EUA = EUA.transpose()
EUA.columns = ['EU ETS']
#plt.figure()

fig, ax= plt.subplots(figsize=(8,7))

EUA.plot(ax = ax, linewidth = 2)
ax.legend(bbox_to_anchor =(0.5,-0.25),  loc='lower center', ncol=1,fontsize=12.5, frameon=False)

ax.set_xticks(range(len(EUA.index)))
ax.xaxis.set_tick_params(labelsize=14,rotation=90)
ax.yaxis.set_tick_params(labelsize=14)
ax.margins(x=0)
ax.grid(axis = 'y', alpha=0.3)
ax.set_ylabel('€/ton',fontsize=15)

fig.savefig('../Figures_Renewables/Energy and climate policy in Europe Fig 1. EU ETS 1 - Price Trajectory.pdf', bbox_inches='tight')

plt.show()

#### Figure 7,GoO graph price trajectory

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
fig.savefig('../Figures_Renewables/Energy and climate policy in Europe Fig 2. GoOs.pdf', bbox_inches='tight')

plt.show()


##############################################################################################################################################################################################

#### LRMCs green vs grey and blue


hygen_lrmc = lrmc_df
hygen_lrmc = lrmc_df.iloc[[18,15,22,10]]
# 18, 15, 22, 10,

index_srmc_blue_base = val_results.index[val_results['params'] == 'SMR_CCS SRMC'].values[0]
srmc_blue = val_results.loc[[index_srmc_blue_base]]
srmc_blue = srmc_blue.set_index('params')

hygen_lrmc = hygen_lrmc.set_index('params')
hygen_lrmc = pd.concat([hygen_lrmc, srmc_blue])
hygen_lrmc.index.name = None
hygen_lrmc = hygen_lrmc.transpose()

hygen_lrmc = hygen_lrmc.rename(columns = {'AEL Flat (50 €/MWh input)': 'AEL Flat (50 €/MWh input) (8322 FLH, WACC 7,5 %)',\
                                          'PEMEL Flexible (15 €/MWh input)': 'PEMEL Flexible (15 €/MWh input) (3000 FLH, WACC 7,5 %)',\
                                          'SOEC Flat (50 €/MWh input)': 'SOEC Flat (50 €/MWh input) (8322 FLH, WACC 7,5 %)',\
                                          'SRMC Grey H2 €/MWh': 'SRMC Grey H2 €/MWh (8322 FLH, WACC 8,7 %)',\
                                          'SMR_CCS SRMC': 'SRMC Blue H2 €/MWh (8322 FLH, WACC 8 %)'})
fig2, ax1 = plt.subplots(figsize = (8,7))
hygen_lrmc.plot()
plt.show()

#### LRMCs for the different hydrogen productions strategis

hygen2 = lrmc_df.iloc[[18,15,22,27,28]]
hygen2 = hygen2.set_index('params')
hygen2.index.name = None
hygen2 = hygen2.transpose()

hygen2 = hygen2.rename(columns = {'AEL Flat (50 €/MWh input)': 'AEL Flat (50 €/MWh input) (8322 FLH, WACC 7,5%)',\
                                          'PEMEL Flexible (15 €/MWh input)': 'PEMEL Flexible (15 €/MWh input) (3000 FLH, WACC 7,5%)',\
                                          'SOEC Flat (50 €/MWh input)': 'SOEC Flat (50 €/MWh input) (8322 FLH, WACC 7,5%)',\
                                          'PEMEL Dedicated (Utility-Solar ContiSouth)': 'PEMEL Dedicated (Utility-Solar ContiSouth) (1785 FLH, WACC 7,5%)',\
                                          'PEMEL Dedicated (Onshore Wind Nordics)': 'PEMEL Dedicated (Onshore Wind Nordics) (3410 FLH, WACC 7,5%)'})



hygen_df = hygen2[['AEL Flat (50 €/MWh input) (8322 FLH, WACC 7,5%)', 
                        'SOEC Flat (50 €/MWh input) (8322 FLH, WACC 7,5%)',
                        'PEMEL Dedicated (Utility-Solar ContiSouth) (1785 FLH, WACC 7,5%)',
                        'PEMEL Dedicated (Onshore Wind Nordics) (3410 FLH, WACC 7,5%)']]
colors = {}
map = ['#9cc674', '#65a528', 'gold', '#0264b1']
for i in range(len(hygen_df.columns)):
    colors[hygen_df.columns[i]] = map[i]
print(colors) 

fig5,ax2 = plt.subplots(figsize = (8,7))

hygen_df.plot(ax = ax2, color = map,linewidth = 2)
hygen2['PEMEL Flexible (15 €/MWh input) (3000 FLH, WACC 7,5%)'].plot(ax=ax2, color = '#9cc674',dashes=[6, 2], linewidth = 2)
ax2.legend(bbox_to_anchor =(0.5,-0.45), loc='lower center', ncol=1,fontsize=13, frameon=False)
ax2.set_xticks(range(len(hygen_df.index)))
ax2.xaxis.set_tick_params(labelsize=14,rotation=90)
ax2.yaxis.set_tick_params(labelsize=14)
ax2.margins(x=0)
ax2.grid(axis = 'y', alpha=0.3)
ax2.set_ylabel('€/MWh',fontsize=15)

fig5.savefig('../Figures_Renewables/Hydrogen Fig 3. LRMCs for the different hydrogen productions strategies.pdf', bbox_inches='tight')

plt.show()


#plt.show()
# 18, 15,22, 27, 28


##### Hydrogen Price

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

fig.savefig('../Figures_Renewables/Hydrogen Fig 7. Hydrogen price.pdf', bbox_inches='tight')


plt.show()


#### Hydrogen Production Growth, StormGeo Forecast
hdrg = pd.read_excel(nordic_growth_pth, sheet_name = "Hdrg Nordics", header = 4)
print(hdrg)

indices = [34,37]
hdrg = hdrg.iloc[indices]
hdrg.columns = hdrg.columns.astype(str)
hdrg = hdrg.drop(columns=['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', '2022', '2023', '2024'])
hdrg['vars'] = ['Annual Hydrogen Production (Flat)', 'Annual Hydrogen Production (Flex)']
hdrg = hdrg.set_index('vars').drop(columns='Hydrogen flat TWh el')
hdrg.index.name = None

remove = ['La stå']
for i in hdrg.columns:
    if 'Unnamed' in i:
        remove.append(i)

hdrg = hdrg.drop(columns=remove)
hdrg = hdrg.transpose()


fig8, ax = plt.subplots(figsize = (8,7))

colormap = ['#183200', '#adc893']
labs = ['Annual Hydrogen Production (Flat)', 'Annual Hydrogen Production (Flex)']
plt.stackplot(hdrg.index, [hdrg['Annual Hydrogen Production (Flat)'], hdrg['Annual Hydrogen Production (Flex)']], alpha = 0.33, colors=colormap, labels=labs)
ax.legend(bbox_to_anchor =(0.5,-0.25), loc='lower center', ncol=1,fontsize=14, frameon=False)
xticks = return_xticks(hdrg.index, ['2025','2030','2035','2040','2045','2050'])
ax.set_xticklabels(xticks,rotation = 90,fontsize=14)
ax.yaxis.set_tick_params(labelsize=14)
ax.margins(x=0)
ax.grid(axis = 'y', alpha=0.3)
ax.set_ylabel('Mt',fontsize=15)

fig8.savefig('../Figures_Renewables/Hydrogen Fig 9. Hydrogen Production Growth.pdf', bbox_inches='tight')

plt.show()
##############################################################################################################################################################################################


### LRMCs of flexible generation technologies

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


fig4.savefig('../Figures_Renewables/Summarizing chapter Fig 3. Flexible Production.pdf', bbox_inches='tight')
fig4.savefig('../Figures_Renewables/Flexible Energy Solutions Fig 1. LRMCs of flexible generation technologies.pdf', bbox_inches='tight')

plt.show()

#### LRMCs - PowerToPower
"""
"""
ptp_lrmc = lrmc_df.iloc[[51,53,49,46]]

ptp_lrmc = ptp_lrmc.set_index('params')
ptp_lrmc.index.name = None
ptp_lrmc = ptp_lrmc.transpose()

ptp_lrmc = ptp_lrmc.rename(columns={'PtP Onshore Wind Dedicatd (Nordics)': 'PtP Onshore Wind Dedicated (Nordics) (3410 FLH, WACC 7.5%)',\
                                      'PtP Ultility-Solar Dedicated (Conti South)': 'PtP Utility-Solar Dedicated (Conti South) (1785 FLH, WACC 7,5 %)',\
                                      'PtP (20 €/MWh input)': 'PtP (20 €/MWh input) (3000 FLH, WACC 7,5%)',\
                                      'PtP (5 €/MWh input)': 'PtP (5 €/MWh input) (3000 FLH, WACC 7,5%)'})



fig9, ax = plt.subplots(figsize = (8,7))

colors = ['steelblue', 'gold', '#183200', '#468110']

ptp_lrmc.plot(ax = ax, color = colors, linewidth = 2)
ax.legend(bbox_to_anchor =(0.5,-0.35), loc='lower center', ncol=1,fontsize=13, frameon=False)
ax.set_xticks(range(len(ptp_lrmc.index)))
ax.xaxis.set_tick_params(labelsize=14,rotation=90)
ax.yaxis.set_tick_params(labelsize=14)
ax.margins(x=0)
ax.grid(axis = 'y', alpha=0.3)
ax.set_ylabel('€/MWh',fontsize=15)

fig9.savefig('../Figures_Renewables/Flexible Energy Solutions Fig 2. LRMCs - PowerToPower.pdf', bbox_inches='tight')
plt.show()
"""
"""

##############################################################################################################################################################################################


solar_lrmc = lrmc_df.iloc[[67,68,73,74,75,76]]

#67,68,73,74,75,76

solar_lrmc = solar_lrmc.set_index('params')
solar_lrmc.index.name = None
solar_lrmc = solar_lrmc.transpose()

solar_lrmc = solar_lrmc.rename(columns={'PV (Roof top) Nordics': 'PV (Roof top) Nordics (959 FLH, WACC 5,7%)',\
                                      'PV (Utility scale) Nordics': 'PV (Utility scale) Nordics (1020 FLH, WACC 6,2 %)',\
                                      'PV (Roof top) Northern Conti': 'PV (Roof top) Northern Conti (1061 FLH, WACC 5,7%)',\
                                      'PV (Roof top) Southern Conti': 'PV (Roof top) Southern Conti (1326 FLH, WACC 5,7%)',\
                                      'PV (Utility scale) Northern Conti':'PV (Utility scale) Northern Conti (1224 FLH, WACC 6,2%)',\
                                      'PV (Utility scale) Southern Conti':'PV (Utility scale) Southern Conti (1785 FLH, WACC 6,2%)'})
solar_roof = solar_lrmc[['PV (Roof top) Nordics (959 FLH, WACC 5,7%)',
                         'PV (Roof top) Northern Conti (1061 FLH, WACC 5,7%)',
                         'PV (Roof top) Southern Conti (1326 FLH, WACC 5,7%)']].copy()

solar_utility = solar_lrmc[['PV (Utility scale) Nordics (1020 FLH, WACC 6,2 %)',
                            'PV (Utility scale) Northern Conti (1224 FLH, WACC 6,2%)',
                            'PV (Utility scale) Southern Conti (1785 FLH, WACC 6,2%)']].copy()

fig9, ax = plt.subplots(figsize = (8,7))
colors = ['gold', 'darkorange', 'maroon']
solar_roof.plot(ax = ax, color = colors, linewidth = 2)
solar_utility.plot(ax = ax, color=colors, linewidth = 2, dashes=[6, 2])

ax.legend(bbox_to_anchor =(0.5,-0.45),  loc='lower center', ncol=1,fontsize=12.5, frameon=False)

ax.set_xticks(range(len(solar_lrmc.index)))
ax.xaxis.set_tick_params(labelsize=14,rotation=90)
ax.yaxis.set_tick_params(labelsize=14)
ax.margins(x=0)
ax.grid(axis = 'y', alpha=0.3)
ax.set_ylabel('€/MWh',fontsize=15)


fig9.savefig('../Figures_Renewables/SolarPower Fig 1. LRMCs Solar Power.pdf', bbox_inches='tight')

plt.show()


##############################################################################################################################################################################################


# LRMCs Onshore Wind Power



wind_lrmc = lrmc_df.iloc[[61,62,70]]

wind_lrmc = wind_lrmc.set_index('params')
wind_lrmc.index.name = None
wind_lrmc = wind_lrmc.transpose()

wind_lrmc = wind_lrmc.rename(columns={'Onshore wind Norway': 'Onshore wind Norway (3511 FLH, WACC 6,2%)',\
                                      'Onshore wind Nordics': 'Onshore wind Nordics (3410 FLH, WACC 6,2 %)',\
                                      'Onshore wind Conti': ' Onshore wind Conti (2909 FLH, WACC 6,2%)'})


fig, ax = plt.subplots(figsize = (8,7))
colors = ['darkseagreen', 'darkgoldenrod','darkolivegreen']
wind_lrmc.plot(ax = ax, color = colors, linewidth = 2)


ax.legend(bbox_to_anchor =(0.5,-0.3),  loc='lower center', ncol=1,fontsize=12.5, frameon=False)

ax.set_xticks(range(len(wind_lrmc.index)))
ax.xaxis.set_tick_params(labelsize=14,rotation=90)
ax.yaxis.set_tick_params(labelsize=14)
ax.margins(x=0)
ax.grid(axis = 'y', alpha=0.3)
ax.set_ylabel('€/MWh',fontsize=15)

fig.savefig('../Figures_Renewables/OnshoreWind Fig 2. LRMC Onshore Wind Power.pdf', bbox_inches='tight')

plt.show()

##############################################################################################################################################################################################

### LRMCs Offshore Wind Power

offshore_lrmc = lrmc_df.iloc[[63,64,65,66,71,72]]

offshore_lrmc = offshore_lrmc.set_index('params')
offshore_lrmc.index.name = None
offshore_lrmc = offshore_lrmc.transpose()

offshore_lrmc = offshore_lrmc.rename(columns={'Offshore wind Nordic West (bottom fixed)': 'Offshore wind Nordic West (bottom fixed) (4509 FLH, WACC 7,5%)',\
                                      'Offshore wind Nordic East (bottom fixed) ': 'Offshore wind Nordic East (bottom fixed) (4158 FLH, WACC 7,5 %)',\
                                      'Offshore wind Nordic West (floating)': 'Offshore wind Nordic West (floating) (4509 FLH, WACC 7,5%)',\
                                        'Offshore wind Nordic East (floating)':'Offshore wind Nordic East (floating) (4158 FLH, WACC 7,5%)',\
                                            'Offshore wind(bottom fixed) Conti ': 'Offshore wind Conti (bottom fixed) (4309 FLH, WACC 7,5%)',\
                                                'Offshore wind (floating) Conti': 'Offshore wind Conti (floating) (4309 FLH, WACC 7,5%)'})

offshore_flo = offshore_lrmc[['Offshore wind Conti (floating) (4309 FLH, WACC 7,5%)',
                              'Offshore wind Nordic East (floating) (4158 FLH, WACC 7,5%)',
                              'Offshore wind Nordic West (floating) (4509 FLH, WACC 7,5%)']].copy()
offshore_bot = offshore_lrmc[['Offshore wind Conti (bottom fixed) (4309 FLH, WACC 7,5%)',
                              'Offshore wind Nordic East (bottom fixed) (4158 FLH, WACC 7,5 %)',
                              'Offshore wind Nordic West (bottom fixed) (4509 FLH, WACC 7,5%)']].copy()

fig, ax = plt.subplots(figsize=(8,7))
colors = ['navy','lightblue', 'steelblue']
offshore_flo.plot(ax = ax, color = colors, linewidth = 2)
offshore_bot.plot(ax = ax, color = colors, linewidth = 2, dashes=[6, 2])

ax.legend(bbox_to_anchor =(0.5,-0.45),  loc='lower center', ncol=1,fontsize=12.5, frameon=False)

ax.set_xticks(range(len(offshore_lrmc.index)))
ax.xaxis.set_tick_params(labelsize=14,rotation=90)
ax.yaxis.set_tick_params(labelsize=14)
ax.margins(x=0)
ax.grid(axis = 'y', alpha=0.3)
ax.set_ylabel('€/MWh',fontsize=15)

fig.savefig('../Figures_Renewables/OffshoreWind Fig 3. LRMCs Offshore Wind Power.pdf', bbox_inches='tight')

plt.show()
################################################################################################################################################################################################

## LRMC Nuclear

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

fig.savefig('../Figures_Renewables/NuclearPower Fig 1. LRMC.pdf', bbox_inches='tight')

plt.show()
"""
################################################################################################################################################################################################

## LRMC Bio CHP
"""

bio_lrmc = lrmc_df.iloc[[88]]

bio_lrmc = bio_lrmc.set_index('params')
bio_lrmc.index.name = None
bio_lrmc = bio_lrmc.transpose()

bio_lrmc = bio_lrmc.rename(columns={'Bio CHP': 'Bio CHP (wood chips) with 25 MWel (5500 FLH, WACC 7,5%)'})

fig, ax = plt.subplots(figsize=(8,7))
bio_lrmc.plot(ax = ax, color = 'darkgoldenrod', linewidth = 2)
ax.legend(bbox_to_anchor =(0.5,-0.25),  loc='lower center', ncol=1,fontsize=12.5, frameon=False)

ax.set_xticks(range(len(bio_lrmc.index)))
ax.xaxis.set_tick_params(labelsize=14,rotation=90)
ax.yaxis.set_tick_params(labelsize=14)
ax.margins(x=0)
ax.grid(axis = 'y', alpha=0.3)
ax.set_ylabel('€/MWh',fontsize=15)

fig.savefig('../Figures_Renewables/Bio-CHP Fig 1. LRMC Bio-CHP.pdf', bbox_inches='tight')

### 
plt.show()
print(bio_lrmc.columns)

"""