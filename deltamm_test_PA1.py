# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 16:11:44 2020

@author: 1
"""

#============================  deltamm()  ==============================
#Function:使用增量度量方法合并或匹配两个字段内已识别特征
import numpy as np
import pandas as pd
import time
import sys
sys.path.append(r'F:\Work\MODE\Submit\Spatialvx_PA1')    #导入的函数路径
import make_SpatialVx_PA1    
import FeatureFinder_test_PA3
import deltammSqCen_test_PA1
import data_pre_PA1


def deltamm(x, p, max_delta, const, fun_type, N, verbose):

    if verbose:
        begin_tiid = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

    if (x['Xlabelsfeature'] == None and x['Ylabelsfeature'] == None):
        try:
            sys.exit(0)
        except:
            print("deltamm: no features to merge/match!")
         
    if (x['Xlabelsfeature'] == None):
        try:
            sys.exit(0)
        except:
            print("deltamm: no verification features present.")
    if (x['Ylabelsfeature'] == None):
        try:
            sys.exit(0)
        except:
            print("deltamm: no model features present.")
    out = x.copy()
    #向字典里追加元素
    out.update ({"match_type": "deltamm", "match_message":"Objects merged/matched based on the Baddeley Delta metric via the deltamm function." })
    #a <- attributes(x)    #将x包含的属性信息赋值给a
    #type <- match.arg(type)
    #未翻译
    if (fun_type == "original"):
        print ("提示：该函数暂未翻译，请执行sqcen函数。")
        #res <- deltammOrig(x = x, p = p, max.delta = max.delta, const = const, verbose = verbose, ...)
    if (fun_type == "sqcen"):
        res = deltammSqCen_test_PA1.deltammSqcen(x = x, p = p, max_delta = max_delta, const = const, N = N, verbose = verbose)
    out.update({'Q': res['Q']})
    out.update({'unmatched':res['unmatched'], 'matches':res['matches'], \
                'merges':res['merges'], 'MergeForce':True, 'class_out':'unmatched'})
    return out

#=============================  Example  ===================================
'''
hold = make_SpatialVx_PA1.makeSpatialVx(X = pd.read_csv("F:\\Work\\MODE\\tra_test\\FeatureFinder\\pert000.csv"), 
                    Xhat = pd.read_csv("F:\\Work\\MODE\\tra_test\\FeatureFinder\\pert004.csv"), 
                    loc = pd.read_csv("F:\\Work\\MODE\\tra_test\\FeatureFinder\\ICPg240Locs.csv"), 
                    thresholds = [0.01, 20.01], projection = True, subset = None, timevals = None, reggrid = True,
                    Map = True, locbyrow = True, fieldtype = "Precipitation", units = ("mm/h"), dataname = "ICP Perturbed Cases", obsname = "pert000", 
                    modelname = "pert004" , q = (0, 0.1, 0.25, 0.33, 0.5, 0.66, 0.75, 0.9, 0.95), qs = None)

look_FeatureFinder = FeatureFinder_test_PA3.featureFinder(Object = hold, smoothfun = "disk2dsmooth", 
                     dosmooth = True, smoothpar = 17, smoothfunargs = None,
                     thresh = 310, idfun = "disjointer", minsize = np.array([1]),
                     maxsize = float("Inf"), fac = 1, zerodown = False, timepoint = 1,
                     obs = 1, model = 1)

#deltamm参数
x = look.copy()
p = 2
max_delta = float("Inf")
const = float("Inf")
fun_type = "sqcen"
N = 701
verbose = False    #R:是否需要打印在屏幕
look_deltamm = deltamm(x = look_FeatureFinder.copy(), p = 2, max_delta = float("Inf"), const = float("Inf"), fun_type = "sqcen", N = 701, verbose = False )
'''















