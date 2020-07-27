# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 14:58:07 2020

@author: 1
"""

#============================  FeatureFinder()  ==============================
#Function:Identify spatial features within a verification set using a threshold-based method.

import matplotlib.pyplot as plt
from numpy.fft import ifftshift
from skimage import measure,color
import numpy as np
import pandas as pd
import cv2 as cv
import sys
sys.path.append(r'F:\Work\MODE\tra_test\makeSpatialVx')    #导入的函数路径
import tra_makeSpatialVx_test    #导入makeSpatialVx函数
import time


start = time.clock()    #开始时间

#连通域标记
def labelsConnection(data):
    labels=measure.label(data,connectivity=2)  #connectivity表示连接的模式，1代表4连通，2代表8连通
    dst=color.label2rgb(labels)  #根据不同的标记显示不同的颜色
    print('regions number:',labels.max()+1)  #显示连通区域块数(从0开始标记)
    return dst

#每个连通域内格点数统计,并且判断格点数量是否在阈值范围内
def propCounts(data_2, minsize, maxsize):
    labels = measure.label(data_2,connectivity=2)
    properties = measure.regionprops(labels)
    valid_label = set()
    nums = []
    numsID = []
    for prop in properties:
        #print(prop.area)
        nums.append(prop.area)    #统计连通域内的格点数
        numsID = list(map(lambda x :x > minsize and x < maxsize, nums))    #比较格点数是否在阈值范围内
        if prop.area > minsize and prop.area < maxsize:
            valid_label.add(prop.label)
        current_bw = np.in1d(labels, list(valid_label)).reshape(labels.shape)
    return nums, numsID, current_bw

#R里面新定义的Nfun函数，求和函数,实际函数功能是统计连通域内面积大小,python用prop.area代替
def newFun(Obj):
    #axis为:1：计算每一行的和，axis为0：计算每一列的和,keepdims为保持原有维度
    return np.sum(np.sum(Obj, axis=0, keepdims=True))
        
def featureFinder(object, smoothfun, dosmooth, smoothpar, smoothfunargs ,thresh ,\
                  idfun ,minsize  ,maxsize ,fac ,zerodown , timepoint ,obs ,model):
    
    #theCall <- match.call()    #调用match函数

    if (np.size(minsize) == 1):
        minsize = np.tile(minsize, 2)        #输入观测和预报数据，因而将最小值和最大值展开为两个数
    if (np.size(maxsize) == 1):    #error:object of type 'float' has no len()
        maxsize = np.tile(maxsize, 2)
    if (len(minsize) != 2):
        try:
            sys.exit(0)
        except:
            print("FeatureFinder: invalid min.size argument.  Must have length one or two.")
    if (len(maxsize) != 2):
        try:
            sys.exit(0)
        except:
            print("FeatureFinder: invalid max.size argument.  Must have length one or two.")
    if (any(minsize) < 1):    #a为list
        try:
            sys.exit(0)
        except:
            print("FeatureFinder: invalid min.size argument.  Must be >= 1.")
    if (any(maxsize) < any(minsize)):
        try:
            sys.exit(0)
        except:
            print("FeatureFinder: invalid max.size argument.  Must be >= min.size argument.")
    a = hold 
    dat = {'X':hold['X'], 'Xhat':hold['Xhat']}    #引用hold里的X,Xhat要素并赋值
    X = dat['X']
    Y = dat['Xhat']
    xdim = a['xdim']
    
    
    if (dosmooth):
        if (np.size(smoothpar) == 1):
            smoothpar = np.tile(smoothpar, 2)    #观测场和预报场的平滑参数一致
        elif (len(smoothpar) > 2):
            try:
                sys.exit(0)
            except:
                print("FeatureFinder: invalid smoothpar argument.  Must have length one or two.")
        
         #调用disk2dsmooth中的kernel2dsmooth卷积平滑，python里面目前用2D卷积平滑替代
        kernel_X = np.ones((smoothpar[0], smoothpar[0]), np.float32)/5    #X的卷积核
        kernel_Y =  np.ones((smoothpar[1], smoothpar[1]), np.float32)/5    #Y的卷积核
        Xsm = cv.filter2D(np.rot90(X, 1), -1, kernel_X)    #对X做2D卷积平滑
        Ysm = cv.filter2D(np.rot90(Y, 1), -1, kernel_Y)    #对Y做2D卷积平滑
        if (zerodown):
             Xsm = np.where(Xsm > 0, Xsm, 0)    #Xsm中大于0的值被0代替
             Ysm = np.where(Ysm > 0, Ysm, 0)    #Ysm中大于0的值被0代替             
    else:
        Xsm = X
        Ysm = Y
    
    if (np.size(thresh) == 1):
        thresh = np.tile(thresh, 2)
    thresh = thresh * fac
    #二值化，首先生成0矩阵，然后将大于阈值的部分赋值为1
    #但是python里面在进行连通域分析的时候已经进行过二值化，不必单独进行二值化？
    #sIx = np.zeros((xdim[0], xdim[1]))
    #sIy = np.zeros((xdim[0], xdim[1]))
    #sIx = np.where(Xsm < thresh[0], Xsm, 1)
    #sIy = np.where(Ysm < thresh[1], Xsm, 1)
    
    #连通域分析
    #连通域分析的时候，分析的是经过模糊处理的图像
    #R里面的阈值为5,2D卷积平滑时增强图像，阈值设为310才能结果对应
    Xfeats = labelsConnection(Xsm > thresh[0])    
    Yfeats = labelsConnection(Ysm > thresh[1])
    if (len(Xfeats) == 0):
        Xfeats = None
    if (len(Yfeats) == 0):
        Yfeats = None
    #如果对连通域的面积大小做限制，则需要执行下面的判断
    if (any(minsize > 1) or any(maxsize < (xdim[0]*xdim[1]))):
        #统计每个连通域内的格点数
        if (Xfeats != None):
            Xnums = propCounts(np.rot90(Xsm > 5, 1))[0]
            XnumsID = propCounts(np.rot90(Xsm > 5, 1))[1]
            Xfeats = propCounts(np.rot90(Xsm > 5, 1))[2]
        if (Yfeats != None):
            Ynums = propCounts(np.rot90(Ysm > 5, 1))[0]
            YnumsID = propCounts(np.rot90(Ysm > 5, 1))[1]
            Yfeats = propCounts(np.rot90(Ysm > 5, 1))[2]
    
    #Xlab = np.zeros(xdim[0], xdim[1])
    #Ylab = np.zeros(xdim[0], xdim[1])
    
    if (np.all(Xfeats) != None):
        Xlab = Xfeats            
    else:
        Xfeats = None
        
    if (np.all(Yfeats) != None):
        Ylab = Yfeats            
    else:
        Yfeats = None    
    
    #返回的out列表里面包括:观测数据（X）,预报数据（Xhat），连通域分析后单个区域（X.feats,Yfeats）并包括了很多的属性信息
    #X.feats,Yfeats包括了很多的属性信息：x,y的范围xrange,yrange，维度dim,步长xstep,ystep,行列信息等；标记后的连通域（Xlab,Ylab）；
    #识别函数、标记函数名称：Convolution Threshold。
    #输出格式与R不完全对应
    out = {'X':hold['X'], 'Xhat':hold['Xhat'], "Xlabeled":Xfeats, "Ylabeled":Yfeats}
    return out
        
        
    
    
    
    
#=============================  Example  ===================================

#make.SpatialVx参数
X = pd.read_csv("F:\\Work\\MODE\\tra_test\\FeatureFinder\\pert000.csv")    #X观测数据
Xhat = pd.read_csv("F:\\Work\\MODE\\tra_test\\FeatureFinder\\pert004.csv")    #Xhat预报数据
thresholds = [0.01, 20.01]    #绘图时colorbar的阈值，函数计算本身没有涉及阈值
loc = pd.read_csv("F:\\Work\\MODE\\tra_test\\FeatureFinder\\ICPg240Locs.csv")    #经纬度信息
projection = True    #是否需要投影
subset = None     #子集
timevals = None
reggrid = True    #logical, is the grid a regular grid?
map = True    #是否需要添加底图
locbyrow = True
fieldtype = "Precipitation"    #标题信息
units = ("mm/h")   #单位
dataname = "ICP Perturbed Cases"     #数据名称
obsname = "pert000"    #观测数据名称
modelname = "pert004"    #模式数据名称
q = (0, 0.1, 0.25, 0.33, 0.5, 0.66, 0.75, 0.9, 0.95)    #所要计算的分位数，函数给定
qs = None


hold = tra_makeSpatialVx_test.makeSpatialVx(X, Xhat, thresholds, loc, projection, subset, timevals,reggrid,\
                     True, locbyrow, fieldtype, units, dataname, obsname, modelname, q, qs)


#FeatureFinder参数
object = hold    #make.SpatialVx生成的结果
smoothfun = "disk2dsmooth"
dosmooth = True
smoothpar = 17    #卷积核的大小
smoothfunargs = None
thresh = 310    #R里面的阈值为5,2D卷积平滑增强图像，阈值设为310才能结果对应
idfun = "disjointer"
minsize = np.array([1])    #判断连通域大小的下限，超过两个值的话用数组
maxsize = float("Inf")    #判断连通域大小的上限，默认无穷大,如果是数值的话，可以直接赋值，超过两个值的话用数组
fac = 1
zerodown = False
timepoint = 1
obs = 1
model = 1

look = featureFinder(object, smoothfun, dosmooth, smoothpar, smoothfunargs,\
                     thresh, idfun, minsize, maxsize, fac, zerodown, timepoint, obs, model)



#绘图
plt.figure(1, dpi = 80)
plt.imshow(look['Xlabeled'])
plt.colorbar(shrink=0.5)

plt.figure(2, dpi = 80)
plt.imshow(look['Ylabeled'])
plt.colorbar(shrink=0.5)

#计算程序运行时间
end = time.clock()
print('Running time: %s s'%(end-start))




















