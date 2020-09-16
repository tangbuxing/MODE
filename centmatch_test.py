# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 15:50:13 2020

@author: 1
"""

#============================  centmatch()  ==============================
#Function:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import sys
sys.path.append(r'F:\Work\MODE\tra_test\makeSpatialVx')
import tra_makeSpatialVx_test
sys.path.append(r'F:\Work\MODE\tra_test\FeatureFinder')
import FeatureFinder_test_PA3


def featureProps(x, whichprops):
    '''
    function (x, Im = NULL, which.props = c("centroid", "area", 
        "axis", "intensity"), areafac = 1, q = c(0.25, 
        0.9), loc = NULL, ...) 
    {
        out <- list()
        if (is.element("centroid", which.props)) {
            if (is.null(loc)) {
                xd <- dim(x$m)
                loc <- cbind(rep(1:xd[1], xd[2]), rep(1:xd[2], each = xd[1]))
            }
            xcen <- mean(loc[c(x$m), 1], na.rm = TRUE)
            ycen <- mean(loc[c(x$m), 2], na.rm = TRUE)
            centroid <- list(x = xcen, y = ycen)
            out$centroid <- centroid
        }
        if (is.element("area", which.props)) 
            out$area <- sum(colSums(x$m, na.rm = TRUE), na.rm = TRUE) * 
                areafac
        if (is.element("axis", which.props)) 
            out$axis <- FeatureAxis(x = x, fac = areafac, ...)
        if (is.element("intensity", which.props)) {
            ivec <- matrix(NA, ncol = length(q), nrow = 1)
            colnames(ivec) <- as.character(q)
            ivec[1, ] <- quantile(c(Im[x$m]), probs = q)
            out$intensity <- ivec
        }
        return(out)
    }
    '''
    
    xd = np.shape(x['_label_image'])    #观测场与预报场的范围大小保持一致。旋转以后的矩阵行列数也变化,xd应为未旋转前的shape
    if ('centroid' in whichprops):
        centroid = xd - np.array(x['centroid'])
    if ('area' in whichprops):
        area = x['area']
    '''
        if ('axis' in whichprops):
            axis = "提示：请参考look['Xprop']或者look['Xprop']里面的'axis'属性"
        if ('intensity' in whichprops):
            intensity = "提示：请参考look['Xprop']或者look['Xprop']里面的'intensity'属性"
    '''
    out = {'centroid':centroid, 'area':area}
    return out

def rdistEarth(x1, x2, miles, R):
    #计算球面距离，这个函数目前没用到
    if (R is None):
        if (miles):
            R = 3963.34
        else :
            R = 6378.388
    coslat1 = math.cos((x1[:, 1] * math.pi)/180)
    sinlat1 = math.sin((x1[:, 1] * math.pi)/180)
    coslon1 = math.cos((x1[:, 0] * math.pi)/180)
    sinlon1 = math.sin((x1[:, 0] * math.pi)/180)
    if (x2 is None):
        cbind = np.mat((coslat1 * coslon1, coslat1 * sinlon1, sinlat1))
        t_cbind = np.mat((coslat1 * coslon1, coslat1 * sinlon1, sinlat1)).T
        pp = np.dot(cbind, t_cbind)
        if (abs(pp) > 1):
            return R * math.acos(1 * np.sign(pp))
        else :
            return R * math.acos(pp)
        
    else :
        coslat2 = math.cos((x2[:, 1] * math.pi)/180)
        sinlat2 = math.sin((x2[:, 1] * math.pi)/180)
        coslon2 = math.cos((x2[:, 0] * math.pi)/180)
        sinlon2 = math.sin((x2[:, 0] * math.pi)/180)
        cbind = np.mat((coslat1 * coslon1, coslat1 * sinlon1, sinlat1))
        t_cbind = np.mat((coslat2 * coslon2, coslat2 * sinlon2, sinlat2)).T
        pp = np.dot(cbind, t_cbind)
        if (abs(pp) > 1):
            return R * math.acos(1 * np.sign(pp))
        else :
            return R * math.acos(pp)
        
def rdist(x1, x2):
    #计算两点间欧式距离
    return np.sqrt(np.sum(np.square(x1 - x2)))
            
'''
    if (class(x) != "features") 
        stop("centmatch: invalid object, x or y type.")
    if (is.null(x$X.feats) && is.null(x$Y.feats)) 
        stop("centmatch: no features to match!")
    if (is.null(x$X.feats)) 
        stop("centmatch: no features in verification field to match.")
    if (is.null(x$Y.feats)) 
        stop("centmatch: no features in model field to match.")
    out <- x
    out$match.message <- "Matching based on centroid distances using centmatch function."
    out$match.type <- "centmatch"
    out$criteria <- criteria
'''
def centmatch(x, criteria, const, distfun, areafac, verbose):
    '''
    #勿删，目前x的属性里面没有加入"features"这个属性
        if (x['class'] != "features"):            
            try:
                sys.exit(0)
            except:
                print("centmatch: invalid object, x or y type.")
        if (x['Xfeature'] == None and x['Yfeature'] == None):
            try:
                sys.exit(0)
            except:
                print("centmatch: no features to match!")
        if (x['Xfeature'] == None):
            try:
                sys.exit(0)
            except:
                print("centmatch: no features in verification field to match!")
        if (x['Yfeature'] == None):
            try:
                sys.exit(0)
            except:
                print("centmatch: no features in model field to match!")
    '''
    '''
        if (criteria == 3) 
            out$const <- const
        else out$const <- NULL
        a <- attributes(x)
    '''
    out = x.copy()
    out.update({'match_message':"Matching based on centroid distances using centmatch function.",\
                      'match_type':"centmatch", 'criteria':criteria})
    if (criteria == 3):
        out.update({'const':const})
    else :
        out.update({'const':str(' NULL ')})
    #获取x的属性，赋值到a
    '''
        if (distfun == "rdist.earth") {
            loc <- a$loc    #直接读取文件，不用引用
            if (is.null(loc)) 
                warning("Using rdist.earth, but lon/lat coords are not available. Can pass them as an attribute to x called loc.")
        }
        else loc <- NULL
        xdim <- dim(x$X.labeled)
        Y <- x$Y.feats
        X <- x$X.feats
        m <- length(Y)
        n <- length(X)
    '''
    if (distfun == "rdist.earth"):
        #loc <- a$loc    #直接读取文件，不用引用（问题：用户在输入数据的时候，需要单独提取经纬度？）
        if (loc == None):
            print("warning:Using rdist.earth, but lon/lat coords are not available. Can pass them as an attribute to x called loc.")
    else :
        loc = None
        xdim = np.shape(x['X'])    #look['Xlabeled']在python里面进行连通域分析以后变为三维，但是数据shape与原来的field没变
        Y = x['Ylabelsfeature']    #x['Ylabelsfeature']对应x$Y.feats：被标记的连通域单独存放
        X = x['Xlabelsfeature']
        m = len(Y)
        n = len(X)
    '''
        if (criteria != 3) {
            Ax <- numeric(n)
            Ay <- numeric(m)
        }
        Dcent <- matrix(NA, m, n)
    '''
    if (criteria != 3):
        Ax = np.zeros((n))
        Ay = np.zeros((m))
    Dcent = np.repeat(None, m*n).reshape(m, n)
    '''
        if (verbose) {
            if (criteria != 3) 
                cat("\n", "Looping through each feature in each field to find the centroid differences.\n")
            else cat("\n", "Looping through each feature in each field to find the areas and centroid differences.\n")
        }
    '''
    if (verbose):
        if (criteria != 3):
            print("\n", "Looping through each feature in each field to find the centroid differences.\n")
        else :
            print("\n", "Looping through each feature in each field to find the areas and centroid differences.\n")
    '''
        for (i in 1:m) {
            if (verbose) 
                cat(i, "\n")
            if (criteria != 3) {
                tmpy <- FeatureProps(x = Y[[i]], which.props = c("centroid", 
                    "area"), areafac = areafac, loc = loc)
                Ay[i] <- sqrt(tmpy$area)
            }
            else tmpy <- FeatureProps(x = Y[[i]], which.props = "centroid", 
                areafac = areafac, loc = loc)
            ycen <- tmpy$centroid
            for (j in 1:n) {
                if (verbose) 
                    cat(j, " ")
                if (criteria != 3) {
                    tmpx <- FeatureProps(x = X[[j]], which.props = c("centroid", 
                      "area"), areafac = areafac, loc = loc)
                    Ax[j] <- sqrt(tmpx$area)
                }
                else tmpx <- FeatureProps(x = X[[j]], which.props = "centroid", 
                    areafac = areafac, loc = loc)
                xcen <- tmpx$centroid
                Dcent[i, j] <- do.call(distfun, c(list(x1 = matrix(c(xcen$x, 
                    xcen$y), 1, 2), x2 = matrix(c(ycen$x, ycen$y), 
                    1, 2), ...)))
            }
            if (verbose) 
                cat("\n")
        }
    '''
    tmpy = {}
    Ay = np.array([])
    ycen = []
    Dcent = np.array([])
    for i in range(m) :
        if (verbose):
            print(i)
        if (criteria != 3):
            #tmpy = featureProps(x = Y['labels_{}'.format(i)], whichprops = ["centroid","area"])
            tmpy_i = featureProps(x = look['Yprop'][i], whichprops = ["centroid","area"])
            tmpy_num = {'{}'.format(i):tmpy_i}
            tmpy.update(tmpy_num)
            Ay_i = math.sqrt(tmpy[str(i)]['area'])
            Ay = np.append(Ay, Ay_i)
        else :
            tmpy = featureProps(x = look['Yprop'][i], whichprops = ["centroid"])
        ycen_i = np.mat(tmpy[str(i)]['centroid'])
        ycen.append(ycen_i)
        
        tmpx = {}
        Ax = np.array([]) 
        xcen = []
        for j in range(n):
            if (verbose):
                print(j)
            if (criteria != 3):
                tmpx_j = featureProps(x = look['Xprop'][j], whichprops = ["centroid","area"])
                tmpx_num = {'{}'.format(j):tmpx_j}
                tmpx.update(tmpx_num)
                Ax_j = math.sqrt(tmpx[str(j)]['area'])
                Ax = np.append(Ax, Ax_j)
            else:
                tmpx = featureProps(x = look['Xprop'][j], whichprops = ["centroid"])
            xcen_j = np.mat(tmpx[str(j)]['centroid'])
            xcen.append(xcen_j)
            #print('length:',len(xcen),'i:',i)
            Dcent_ij = rdist(x1 = xcen[j], x2 = ycen[i])    #miles/R为默认参数
            Dcent = np.append(Dcent, Dcent_ij)      
        if (verbose):
            print('\n')
    Dcent = Dcent.reshape(m, n)
    '''
        if (criteria != 3) {
        Ay <- matrix(rep(Ay, n), m, n)
        Ax <- matrix(rep(Ax, m), m, n, byrow = TRUE)
        }
        if (criteria == 1) 
            Dcomp <- Ay + Ax
        else if (criteria == 2) 
            Dcomp <- (Ax + Ay)/2
        else if (criteria == 3) 
            Dcomp <- matrix(const, m, n)
        else stop("centmatch: criteria must be 1, 2 or 3.")
        DcompID <- Dcent < Dcomp
        any.matched <- any(DcompID)
        FobjID <- matrix(rep(1:m, n), m, n)
        OobjID <- t(matrix(rep(1:n, m), n, m))
        fmatches <- cbind(c(FobjID)[DcompID], c(OobjID)[DcompID])
        colnames(fmatches) <- c("Forecast", "Observed")
    '''
    if (criteria != 3):
        Ay = np.repeat(Ay, n).reshape(m, n)
        Ax = np.tile(Ax, m).reshape(m, n)
    if (criteria == 1):
        Dcomp = Ay + Ax
    elif (criteria == 2):
        Dcomp = (Ay + Ax)/2
    elif (criteria == 3):   
        Dcomp = np.repeat(const, n*m ).reshape(m, n)
    else:
        try:
            sys.exit(0)
        except:
            print("centmatch: criteria must be 1, 2 or 3.")
    DcompID = Dcent < Dcomp    #Dcent为两点间的欧式距离，Dcomp = 观测场面积的开方+预报场面积的开方
    any_matched = np.any(DcompID)
    FobjID = np.repeat(np.arange(m),n).reshape(m, n)
    OobjID = np.tile(np.arange(n),m).reshape(n, m)
    fmatches = np.argwhere(DcompID)    #获取True的索引
    #colnames(fmatches) <- c("Forecast", "Observed")
    '''
        if (dim(fmatches)[1] > 1) {
        pcheck <- paste(fmatches[, 1], fmatches[, 2], sep = "-")
        dupID <- duplicated(pcheck)    #判断是否存在重复，并来舍弃重复的
        if (any(dupID)) 
            fmatches <- fmatches[!dupID, , drop = FALSE]
        oID <- order(fmatches[, 1])
        fmatches <- fmatches[oID, , drop = FALSE]
    }
    if (is.null(dim(fmatches)) && length(fmatches) == 2) 
        fmatches <- matrix(fmatches, ncol = 2)
    out$matches <- fmatches
    '''
    
    if (len(fmatches[:, 0]) > 1):
        pcheck = []
        for k in range(len(fmatches[:, 0])):
            pcheck_k = '{0} - {1}'.format(fmatches[k-1, 0], fmatches[k-1, 1])
            pcheck.append(pcheck_k)
        dupID = set(pcheck)    #集合自动去重，但是原函数是判断是否重复后，从fmatches里面删除重复
    if (len(dupID) != len(fmatches)):
        print("提示：存在一模一样的配对，需要从fmatches里面删除。")    
    if (np.shape(fmatches) is None and len(fmatches) == 2):
        fmatches = np.mat(fmatches)
    out.update({'matches':fmatches})
    '''
        if (any.matched) {
        funmatched <- (1:m)[!is.element(1:m, fmatches[, 1])]    #比较1:m与fatches的大小，返回
        vxunmatched <- (1:n)[!is.element(1:n, fmatches[, 2])]
        matchlen <- dim(fmatches)[1]
        fuq <- unique(fmatches[, 1])
        flen <- length(fuq)
        ouq <- unique(fmatches[, 2])
        olen <- length(ouq)
        if (matchlen == flen && matchlen > olen) {
            if (verbose) 
                cat("Multiple observed features are matched to one or more forecast feature(s).  Determining implicit merges.\n")
        }
        else if (matchlen > flen && matchlen == olen) {
            if (verbose) 
                cat("Multiple forecast features are matched to one or more observed feature(s).  Determining implicit merges.\n")
        }
        else if (matchlen > flen && matchlen > olen) {
            if (verbose) 
                cat("Multiple matches have been found between features in each field.  Determining implicit merges.\n")
        }
        else if (matchlen == flen && matchlen == olen) {
            if (verbose) 
                cat("No multiple matches were found.  Thus, no implicit merges need be considered.\n")
        }
        implicit.merges <- MergeIdentifier(fmatches)
    }
    else {
        if (verbose) 
            cat("No objects matched.\n")
        implicit.merges <- NULL
        funmatched <- 1:m
        vxunmatched <- 1:n
    }
    out$unmatched <- list(X = vxunmatched, Xhat = funmatched)
    out$implicit.merges <- implicit.merges
    out$criteria.values <- Dcomp
    out$centroid.distances <- Dcent
    out$MergeForced <- FALSE
    class(out) <- "matched"
    return(out)}
    '''
    if (any_matched):
        f_comped = set(np.arange(m))
        funmatched = f_comped - set(fmatches[:, 0])     #返回观测场中没有匹配到的object
        vxunmatched = f_comped - set(fmatches[:, 1])    #返回预报场中没有匹配到的object
        matchlen = len(fmatches[:,0])
        fuq = set(fmatches[:, 0])     #使得fmatches第一列没有重复的值
        flen = len(fuq)
        ouq = set(fmatches[:, 1])
        olen = len(ouq)
        if (matchlen == flen and matchlen > olen):    #注意：这一步的结果与R语言判断有出入
            if (verbose):
                print("Multiple observed features are matched to one or more forecast feature(s).  Determining implicit merges.\n")
        elif (matchlen > flen and matchlen > olen):
            if (verbose):
                print("Multiple forecast features are matched to one or more forecast feature(s).  Determining implicit merges.\n")
        elif (matchlen > flen and matchlen > olen):
            if (verbose):
                print("Multiple matches have been found between features in each field.  Determining implicit merges.\n")
        elif (matchlen == flen and matchlen == olen):
            if (verbose):
                print("No multiple matches were found.  Thus, no implicit merges need be considered.\n")
        implicit_merges = fmatches    #R:调用MergeIdentifier
    else :
        if (verbose):
            print("No objects matched.\n")
        implicit_merges = None
        funmatched = np.arange(m)
        vxunmatched = np.arange(n)
    matched = {'X':vxunmatched, 'Xhat':funmatched}
    out.update({'matched':matched, 'implicit_merges':implicit_merges, \
                      'criteria_values':Dcomp, 'centroid_distances':Dcent, \
                      'MergeForced':False, 'class(out)':"matched"})

            
    return out

hold = tra_makeSpatialVx_test.makeSpatialVx(X = pd.read_csv("F:\\Work\\MODE\\tra_test\\makeSpatialVx\\UKobs6.csv"), \
                    Xhat = pd.read_csv("F:\\Work\\MODE\\tra_test\\makeSpatialVx\\UKfcst6.csv"), \
                    loc = pd.read_csv("F:\\Work\\MODE\\tra_test\\makeSpatialVx\\UKloc.csv"), \
                    thresholds = [0.01, 20.01], projection = True, subset = None, timevals = None, reggrid = True,\
                    Map = True, locbyrow = True, fieldtype = "Precipitation", units = ("mm/h"), dataname = "ICP Perturbed Cases", obsname = "pert000", \
                    modelname = "pert004" , q = (0, 0.1, 0.25, 0.33, 0.5, 0.66, 0.75, 0.9, 0.95), qs = None)

look = FeatureFinder_test_PA3.featureFinder(Object = hold, smoothfun = "disk2dsmooth", \
                     dosmooth = True, smoothpar = 17, smoothfunargs = None,\
                     thresh = 100, idfun = "disjointer", minsize = np.array([1]),\
                     maxsize = float("Inf"), fac = 1, zerodown = False, timepoint = 1,\
                     obs = 1, model = 1)

#centmatch参数：
#经纬度信息
'''
loc = pd.read_csv("F:\\Work\\MODE\\tra_test\\makeSpatialVx\\UKloc.csv")
x = look
criteria = 1
const = 14
distfun = "rdist"
areafac = 1
verbose = False
'''

look2 = centmatch(x = look, criteria = 1, const = 14, distfun = "rdist", \
                  areafac = 1, verbose = False)






