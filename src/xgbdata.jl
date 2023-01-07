# Structure for passing labels and weights along with DataFrame to DMatrix
using XGBoost

"""
    XGBData

    This is a mutable structure that facilitates creation of DMatrix needed for XGBOOST.
    It wraps a DataFrame with a target name (i.e. ylabel) ortarget vector (i.e. label).  
    Class weights can be added as either a vector (i.e. weight) or
    a class specific designation (i.e. classweights).  A special condition is
    classweights=[0.0]. This generates a classweight vector with classweights inverse of
    class proportion and normalized to most frequent class.

    Example
    ```
    mytrainer=XGBData(traindf,ylabel="myTarget")
    mytrainer2=XGBData(traindf2,ylabel2="someTarget", classweights=[1.0, 1.8])
    ```


"""
mutable struct XGBData
    xdata::DataFrame
    ylabel::String
    label::Vector{Float64}
    weight::Vector{Float64}
    classweights::Vector{Float64}
    XGBData(xdata::DataFrame ;
            ylabel::String="", label::Vector{Float64}=[-1.0],
            weight::Vector{Float64}=[-1.0],
            classweights::Vector{Float64}=[-1.0])=new(xdata,ylabel,label,weight,classweights)
end

function XGBoost.DMatrix(xD::XGBData)
    # function to convert XGBData to DMatrix
    # XGBoost.getinfo(someDMatrix,Float64,"label")
    lflag=wflag=0
    nr=size(xD.xdata)[1]
    if length(xD.label)==nr
        lflag=1
    elseif xD.ylabel != ""
        lflag=2
    end
    if length(xD.weight)==nr
        wflag=1
    elseif xD.classweights != [-1.0]
        wflag=2
    end
    if lflag==0 && wflag==0
        return DMatrix(xD.xdata)
    end
    xddata=copy(xD.xdata)
    xdlabel=Vector{Float64}(undef,0)
    if lflag==1
        xdlabel=xD.label
    elseif lflag==2
        xdlabel=xD.xdata[!,Symbol(xD.ylabel)]
        select!(xddata, Not(Symbol(xD.ylabel)))
    end
    if wflag==1
        xdweight=xD.weight
    elseif wflag==2 && lflag>0
        # generate and/or assign  class weights
        mclasses=unique(sort(xdlabel))
        nclass=length(mclasses)
        ydict=Dict{Number,Int64}()
        for i in 1:nclass
            ydict[mclasses[i]]=i
        end
        classweights=xD.classweights
        if classweights==[0.0]
            classweights=ones(nclass)
            yct=countmap(xdlabel)
            yctmax=maximum(collect(values(yct)))
            for ks in keys(yct)
                classweights[ydict[ks]]= yctmax / yct[ks]
            end
        end
        nweights=length(classweights)
        if nweights != nclass
           msg="number of class wts does not match number of classes - can not convert to DMatrix"
           error(msg) 
        end
        nys=length(xdlabel)
        xdweight=Vector{Union{Int64,Float64}}(undef,nys)
        for i in 1:nys
            xdweight[i]=classweights[ydict[xdlabel[i]]]
        end
    end
    if lflag>0 && wflag>0
        return DMatrix(xddata,label=xdlabel,weight=xdweight)
    end
    if lflag>0
        return DMatrix(xddata,label=xdlabel)
    end
    if wflag>0
        return DMatrix(xddata,weight=xdweight)
    end

    msg="XGBData could not convert to DMatrix"
    error(msg)
end
