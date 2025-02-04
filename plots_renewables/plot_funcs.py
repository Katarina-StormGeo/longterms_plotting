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

from file_reads import *
from colormap import *

path_ic = parse_cfg('file_paths.cfg')['input_cost']


path_fe = parse_cfg('file_paths.cfg')['forward_exchange']


# go to Forward exchange rates_Longterms_Fall_2015.xlsx and make sure that the indices are correct
nordics = renewable_growth(path_fe, 'Nordics', [3,4,5,6,6,7], header=1)
norway = renewable_growth(path_fe, 'Norway', [66,67,68,69,70,71,72])
sweden = renewable_growth(path_fe, 'Sweden', [70,71,72,73,74,75,76])
denmark = renewable_growth(path_fe, 'Denmark', [55,56,57,58,59,60,61])
finland = renewable_growth(path_fe, 'Finland', [37,38,39,40,41,42,43])

def nordic_renewable_growth():
    nordics = renewable_growth(path_fe, 'Nordics', [3,4,5,6,6,7], header=1)

    x = nordics.index
    y = []

    for i in nordics.columns:
        y.append(nordics['{}'.format(i)])

    color_map = []
    
    return y

print(nordic_renewable_growth())




