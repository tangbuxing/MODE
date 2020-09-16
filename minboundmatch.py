import time
import sys
import numpy as np
import pandas as pd
from scipy import ndimage
sys.path.append(r'F:\Work\MODE\tra_test\makeSpatialVx')
import tra_makeSpatialVx_test
sys.path.append(r'F:\Work\MODE\tra_test\FeatureFinder')
import FeatureFinder_test_PA3

def minsepfun(Id, dm0, dm1, indX, indXhat):
    #掩膜提取函数：data为原始数值数组，mask为获得掩码用的布尔数组
    '''
        minsepfun <- function(id, dm0, dm1, indX, indXhat) {
        i = id[1]
        j = id[2]
        Obs = min((as.matrix(dm0[[i]]))[as.logical(as.matrix(indXhat[[j]]))], 
            na.rm = TRUE)
        Fcst = min((as.matrix(dm1[[j]]))[as.logical(as.matrix(indX[[i]]))], 
            na.rm = TRUE)
        return (min(c(Fcst, Obs), na.rm = TRUE))
    '''
    result_value = []
    Obs_value = []
    Fcst_value = []
    for i in range(len(Id[:, 0])):
        a = Id[i, 0]    #从id数组的第一列取值
        b = Id[i, 1]    #从id数组的第二列取值
        #print(a, b)
        Obsdata_mask = np.mat(dm0["labels_{}".format(a)])    #需要掩膜的原始数据
        Obs_mask = indXhat["labels_{}".format(b)] < 1     #掩膜范围
        Obs_masked = np.ma.array(Obsdata_mask, mask = Obs_mask)    #返回值有三类masked_array，mask,fill_value
        Obs = np.min(Obs_masked)
        Obs_value.append(Obs)
        
        Fcstdata_mask = np.mat(dm1["labels_{}".format(b)])
        Fcst_mask = indX["labels_{}".format(a)] < 1
        Fcst = np.min(np.ma.array(Fcstdata_mask, mask = Fcst_mask))
        Fcst_value.append(Fcst)
        
        result = min(Fcst, Obs)
        result_value.append(result)
    return result_value

def minboundmatch(x, fun_type, mindist = float('inf'), verbose = False):
    '''
    if (verbose) 
        begin.tiid <- Sys.time()
    if (class(x) != "features") 
        stop("minboundmatch: invalid x argument.")
    if (is.null(x$X.feats) && is.null(x$Y.feats)) 
        stop("minboundmatch: no features to match!")
    if (is.null(x$X.feats)) 
        stop("minboundmatch: no verification features to match.")
    if (is.null(x$Y.feats)) 
        stop("minboundmatch: no model features to match.")
    type <- tolower(type)
    type <- match.arg(type)
    out <- x
    a <- attributes(x)
    out$match.type <- "minboundmatch"
    out$match.message <- paste("Matching based on minimum boundary
    '''    
    if verbose:
        begin_tiid = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    if (type(x) != dict):    #R判断x是否为features,python 暂时判断x是否为dict
        try:
            sys.exit(0)
        except:
            print("minboundmatch: invalid x argument.")
    if (look['Xlabelsfeature'] == None and look['Ylabelsfeature'] == None):
        try:
            sys.exit(0)
        except:
            print("minboundmatch: no features to match!")
    if (look['Xlabelsfeature'] == None):
        try:
            sys.exit(0)
        except:
            print("minboundmatch: no verification features to match.")        
    if (look['Ylabelsfeature'] == None):
        try:
            sys.exit(0)
        except:
            print("minboundmatch: no model features to match.")     
    fun_type = fun_type.lower()     # str.lower(),将type中的字母变为小写
    #type <- match.arg(type)    #未译
    out = x.copy()
    #a <- attributes(x)    #获取x中的属性
    out.update({'match_type ':"minboundmatch"})
    match_message = {'match_message_1' :"Matching based on minimum boundary separation using {} match".format(fun_type[0]),\
                 'match_message_2' :"Matching based on minimum boundary separation using {} match".format(fun_type[1])}
    out.update({'match_message':match_message})
    '''
        if Type == 's':
            Type = "single"
        elif Type == 'm':
            Type = "multiple"
    '''
    Xfeats = x['Xlabelsfeature']
    Yfeats = x['Ylabelsfeature']
    '''
    Xfeats = x$X.feats
    Yfeats = x$Y.feats
    if (!is.null(Xfeats)) 
        n <- length(Xfeats)
    else n <- 0
    if (!is.null(Yfeats)) 
        m <- length(Yfeats)
    else m <- 0
    if (m == 0 && n == 0) {
        if (verbose) 
            cat("\n", "No features detected in either field.  Returning NULL.\n")
        return(NULL)
    }
    else if (m == 0) {
        if (verbose) 
            cat("\n", "No features detected in forecast field.  Returning NULL.\n")
        return(NULL)
    }
    else if (n == 0) {
        if (verbose) 
            cat("\n", "No features detected in observed field.  Returning NULL.\n")
        return(NULL)
    }
    ind <- cbind(rep(1:n, m), rep(1:m, each = n))
    Xdmaps = lapply(Xfeats, distmap, ...)
    Ydmaps = lapply(Yfeats, distmap, ...)
    '''
    if Xfeats != {}:
        n = len(Xfeats)
    else:
        n = 0
    if Yfeats != {}:
        m = len(Yfeats)
    else:
        m = 0
    if (m == 0 and n == 0):
        if verbose:    
            print("\n",'No features detected in either field.  Returning NULL.\n')
        return None
    elif (m == 0):
        if verbose:
            print("\n",'No features detected in forecast field.  Returning NULL.\n')
        return None
    elif (n == 0):
        if verbose:
            print("\n", "No features detected in observed field.  Returning NULL.\n")
        return None
    rep00_1 = np.arange(1, n + 1, 1)
    rep00 = np.tile(rep00_1, m)
    rep0_1 = np.arange(1, m + 1, 1)
    rep0 = rep0_1.repeat(n, axis = 0)    #按行进行元素重复
    ind = np.vstack((rep00, rep0)).T    #数组转置
    
    Xdmaps = {}
    Ydmaps = {}
    for i in range(1, n + 1):
        xdmaps = ndimage.morphology.distance_transform_edt(1 - Xfeats['labels_%d'%i])
        Xdmaps['labels_%d'%i] = xdmaps
    for i in range(1, m + 1):
        ydmaps = ndimage.morphology.distance_transform_edt(1 - Yfeats['labels_%d'%i])
        Ydmaps['labels_%d'%i] = ydmaps
    
    '''
    res <- apply(ind, 1, minsepfun, dm0 = Xdmaps, dm1 = Ydmaps, 
        indX = Xfeats, indXhat = Yfeats)
    res <- cbind(ind, res)
    colnames(res) <- c("Observed Feature No.", "Forecast Feature No.", 
        "Minimum Boundary Separation")
    good <- res[, 3] <= mindist
    res <- res[good, , drop = FALSE]
    out$values <- res
    o <- order(res[, 3])
    res <- res[o, , drop = FALSE]
    '''
    res = minsepfun(Id = ind, dm0 = Xdmaps, dm1 = Ydmaps, indX = Xfeats, indXhat = Yfeats)
    #o = res.rank()
    res = np.column_stack((ind,res))
    res = pd.DataFrame(res, columns = ["Observed Feature No.", "Forecast Feature No.", "Minimum Boundary Separation"])
    #good = res["Minimum Boundary Separation"] <= mindist    #判断计算结果是否小于设置的参数mindist
    #res <- res[good, , drop = False]    #如果res中存在异常值，删除
    res = res[res["Minimum Boundary Separation"] <= mindist].reset_index(drop = True)
    out.update({'values': res})
    
    o = res["Minimum Boundary Separation"].rank()    #dataframe排序，不改变数值的位置，标记排名
    res = res.sort_values(by = "Minimum Boundary Separation").reset_index(drop = True)    #dataframe排序，默认升序，并更新index
    '''
    if (type == "single") {
        N <- dim(res)[1]
        id <- 1:N
        id <- id[o]
        matches <- cbind(numeric(0), numeric(0))
        for (i in 1:N) {
            matches <- rbind(matches, res[1, 2:1])
            id2 <- (res[, 1] == res[1, 1]) | (res[, 2] == res[1, 
                2])
            res <- res[!id2, , drop = FALSE]
            id <- id[!id2]
            if (length(id) == 0) 
                break
        }
    }
    '''
    if (fun_type == "single"):
        N = len(res)
        Id = np.arange(N)
        Id = o
        matches = res[res["Observed Feature No."] == res["Forecast Feature No."]].reset_index(drop = True)
        '''
        #逻辑走得通，但是没循环到
        matches = np.zeros((1,2))
        for i in N:
            matches = res[:1]    #选取res的第一行，也是值最小的配对
            res = res.reset_index(drop = True)     #res排序后，更新index，并且原来的index值不保存
            Id2 = res[(res["Observed Feature No."] == res["Observed Feature No."][0]) | \
                    (res["Forecast Feature No."] == res["Forecast Feature No."][0])]
            Id = Id2[Id2 != True].index    #Id2[Id2 != True].index.tolist #获取index值
            if len(Id) == 0 :
                break
        '''
    else:
        '''
    else {
        matches <- res[, 2:1, drop = FALSE]
        matchlen <- dim(matches)[1]
        fuq <- unique(matches[, 1])
        flen <- length(fuq)
        ouq <- unique(matches[, 2])
        olen <- length(ouq)
        if (matchlen > 0) {
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
            out$implicit.merges <- MergeIdentifier(matches)
        }
        else {
            if (verbose) 
                cat("No objects matched.\n")
            out$implicit.merges <- NULL
        }
    }
        '''
        matches = res[["Observed Feature No.","Forecast Feature No."]]    #获取dataframe前两列
        matchlen = matches.shape[0]
        fuq =  set((matches["Forecast Feature No."]))   #预报场set存放,使得元素不重复
        flen = len(fuq)
        ouq = set((matches["Observed Feature No."]))    #观测场set存放,使得元素不重复
        olen = len(ouq)
        if (matchlen > 0):
            if (matchlen == flen and matchlen > olen):
                if (verbose):
                    print("Multiple observed features are matched to one or more forecast feature(s).  Determining implicit merges.\n")
            elif (matchlen > flen and matchlen == olen):
                if (verbose):
                    print("Multiple forecast features are matched to one or more observed feature(s).  Determining implicit merges.\n")
            elif (matchlen > flen and matchlen > olen):
                if (verbose):
                    print("Multiple matches have been found between features in each field.  Determining implicit merges.\n")
            elif (matchlen == flen and matchlen == olen):
                if (verbose):
                    print("No multiple matches were found.  Thus, no implicit merges need be considered.\n")
        else :
            if (verbose):
                print("No objects matched.\n")
            out.update({"implicit_merges": None})
    
    matches = matches.sort_values(by = ["Forecast Feature No."])    #R：将Forecast放在第一列，并以Forecast进行升序排列，python里面Forecast依然放在第二列
    #matches.insert(0, "Forecast Feature No.", matches.pop("Forecast Feature No."))    #先删除"Forecast Feature No."列，然后在原表中第0列插入被删掉的列,并重新命名
    #matches = pd.DataFrame(matches, columns = ["Forecast", "Observed"])    #R:res[, 2:1]表示先去第二列，再取第一列，所以重新命名，python里面没有调换顺序
    out.update({"matches":matches})
    unmatched_X = set(ind[:, 0]) - set(matches["Observed Feature No."])    #判断ind的值是否在matches里面，输出ind不在matches里面的值
    unmatched_Xhat = set(ind[:, 1]) - set(matches["Forecast Feature No."])
    unmatched = {"X":unmatched_X, "Xhat":unmatched_Xhat}
    out.update({"unmatched": unmatched})
    if (verbose):
        print(begin_tiid,'- begin.tiid')
    out.update({"MergeForced": False})
    out.update({"class":"matched"})
        
    return out


#=============================  Example  ===================================
hold = tra_makeSpatialVx_test.makeSpatialVx(X = pd.read_csv("F:\\Work\\MODE\\tra_test\\makeSpatialVx\\UKobs6.csv"),\
                     Xhat = pd.read_csv("F:\\Work\\MODE\\tra_test\\makeSpatialVx\\UKfcst6.csv"),\
                     thresholds = [0.01, 20.01], loc = pd.read_csv("F:\\Work\\MODE\\tra_test\\makeSpatialVx\\UKloc.csv"),\
                     projection = True, subset = None, timevals = None, reggrid = True, \
                     Map = True, locbyrow = True, fieldtype = "Precipitation", units = ("mm/h"), \
                     dataname = "ICP Perturbed Cases", obsname = "pert000" ,modelname = "pert004",\
                     q = (0, 0.1, 0.25, 0.33, 0.5, 0.66, 0.75, 0.9, 0.95) ,qs = None)

look = FeatureFinder_test_PA3.featureFinder(Object = hold, smoothfun = "disk2dsmooth", \
                     dosmooth = True, smoothpar = 17, smoothfunargs = None,\
                     thresh = 100, idfun = "disjointer", minsize = np.array([1]),\
                     maxsize = float("Inf"), fac = 1, zerodown = False, timepoint = 1,\
                     obs = 1, model = 1)

x = look    #FeatureFinder函数结果
#fun_type = str(["single", "multiple"])     
fun_type = "single"    #类型只能为"single"或者 "multiple"，如果是single,matches结果为一对一（长度为5），multiple的结果为一对多（长度为5*5）
mindist = float('Inf')
verbose = False

look2 = minboundmatch(x = look, fun_type = "single", mindist = float('inf'), verbose = False)
'''
if __name__ == '__main__':
    x = np.load("look.npy",allow_pickle=True)
    minboundmatch(x,Type = 'm',mindist = 2)
'''

