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

def parse_cfg(cfgfile):

    # Reads the config file, "file_paths.cfg"

    with open(cfgfile, 'r') as ymlfile:
        cfgstr = yaml.full_load(ymlfile)

    
    return cfgstr

start_year = parse_cfg('file_paths.cfg')['start_year']

def input_cost(path, sheet_name, valname_or_row,header =0):
    
    sheet = pd.read_excel(path, sheet_name = sheet_name, header = header)
    sheet.index = sheet.index + header + 2
    sheet.columns = sheet.columns.astype(str)
    
    remove = []

    for i in sheet.columns:
        try:
            if int(i) < int(start_year):
                remove.append(i)
        except:
            if i != 'params':
                remove.append(i)
    
    sheet = sheet.drop(columns = remove)
    
    if type(valname_or_row[0]) == int:
        rows = valname_or_row
        df = sheet.loc[rows]
        df = df.set_index('params')
        df.index.name = None

    else:
        valname = valname_or_row
        df = sheet.set_index('params').loc[valname]
        df.index.name = None

    return df.transpose()


def renewable_growth(path, sheet_name, rows, header =4):
    start_year = parse_cfg('file_paths.cfg')['start_year']
    sheet = pd.read_excel(path, sheet_name = sheet_name, header = header)
    sheet.index = sheet.index + header + 2
    sheet.columns = sheet.columns.astype(str)
    sheet.columns.values[0] = 'params'
    remove = []

    for i in sheet.columns:
        
        try:
            if int(i) < int(start_year):
                remove.append(i)
        except:
            if 'params' not in str(i):
                remove.append(i)
    
    sheet = sheet.drop(columns = remove)

    df = sheet.loc[rows].set_index('params')
    df.index.name = None
    #df = df.transpose()[start_year:'2050']
    return df.transpose()



path_fe = parse_cfg('file_paths.cfg')['forward_exchange']


# go to Forward exchange rates_Longterms_Fall_2015.xlsx and make sure that the indices are correct
nordics = renewable_growth(path_fe, 'Nordics', [3,4,5,6,6,7], header=1)
norway = renewable_growth(path_fe, 'Norway', [66,67,68,69,70,71,72])
sweden = renewable_growth(path_fe, 'Sweden', [70,71,72,73,74,75,76])
denmark = renewable_growth(path_fe, 'Denmark', [55,56,57,58,59,60,61])
finland = renewable_growth(path_fe, 'Finland', [37,38,39,40,41,42,43])


#print(finland)