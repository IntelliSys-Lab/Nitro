import numpy as np
import torch
import time
import csv
import collections
import logging
import config
import utils
import ray
from env import Environment
import config
from statsmodels.stats.weightstats import DescrStatsW


if __name__ == "__main__":
    t = [1, 100, 100, 100, 250, 250, 250, 250, 250]
    dic = {}
    for v in t:
        if v not in dic:
            dic[v] = 1
        else:
            dic[v] = dic[v] + 1
        
    value = list(dic.keys())
    density = [x/sum(dic.values()) for x in list(dic.values())]

    print(value)
    print(density)

    dsw = DescrStatsW(value, density)
    cv = dsw.std / abs(dsw.mean)

    print(cv)
    print(utils.cv(t))