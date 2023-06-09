# -*-coding:utf-8 -*-
"""
# File       : ttt.py
# Time       ：2023/6/9 17:52
# version    ：python 3.7
# Description：
"""

import streamlit as st
import pandas as pd
from fastai.tabular.all import *
from fastai.collab import *


def load_data():
    data_df = pd.read_excel('[final] April 2015 to Nov 30 2019 - Transformed Jester Data - .xlsx',
                            header=None,
                            names=None,
                            index_col=None)

    jokes_df = pd.read_excel('Dataset4JokeSet.xlsx',
                             header=None,
                             names=None,
                             index_col=None)

    return data_df, jokes_df


