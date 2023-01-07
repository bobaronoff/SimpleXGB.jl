# routine(s) to fit an xgbmodel. If supplied with test data, metrics are calculated.
# variable importance is reported. partial dependence data is calculated.
using DataFrames
using Statistics
using Distributions
using GLM
using Plots
using Loess

"""
    xgboost_fit()

    This function will fit an XGBOOST model. The thrust of this function is to
    provide objective specific model evaluation.  Objectives supported include
    reg:squarederror, binary:logisitc, and mult:softprob.

    Traindata can be any form that can be implicitly converted to a DMatrix type.
    The preferred input type is 'XGBData' which holds weight data.  This type is 
    required to produce partial dependence plots. Alternate data types can
    utilize a class weight matrix allows assignment of weight by class.  Classification
    objectives (i.e. binary:logistic,multi:softprob) allow specification of a loss
    matrix (column heads reflect predicted and row numbers reflect true).

    Variable importance report can be controlled with 'importancecontrol'. A fraction
    represents the top percentage total gain included.  A whole number reflects the 
    top-n variables.

    Standard XGBOOST parameters are entered as keyword parameters.

    Results are returned as a named tuple:
        booster => the result xgboost model
        params => a dictionary of xgboost parameters
        metrics => a dictionary of objective specific metrics
        variables => a DataFrame with gain and cover model metrics
        pdp => a Dictionary with each key corresponding to a reported variable.
                each vaule is a named tuple:
                                            plot => partial dependence plot
                                            x => x value of plot
                                            y => y value of plot
                                            misspdp => value of missing
                                            meanyhat => mean of variable
        plots => vector of all plots                                    

    Example:

    ```
    mytrainer=XGBData(traindf, ylabel="Y", classweights=[1.0,1.8]) 
    mytester=XGBData(testdf, ylabel="Y") 
    myfit=xgboost_fit(mytrainer, testdata=mytester, modelname="MyModel",
                        costmatrix= [ 0.0 .75 ; 0.0 1.5]),
                        importancecontrol=.9,
                        objective="binary:logistic",
                        num_round=500, eta=0.01, subsample=0.5, 
                        max_depth=4,gamma=5);

    ```


"""
function xgboost_fit(traindata; 
                        testdata::Any=[],
                        modelname::String="",
                        showplots::Bool=true,
                        costmatrix=[1.0],
                        classweights=[-1.0],
                        importancecontrol::Union{Float64,Int64}=-1,
                        kw... )

    params=(
            modelname=modelname,
            showplots=showplots,
            costmatrix=costmatrix,
            classweights=classweights,
            importancecontrol=importancecontrol,
            kw=kw  )
    validobjectives=["reg:squarederror","binary:logistic","multi:softprob"]
    # convert kw... to Dict(); reverse kw=(;kwdict...)
    kwdict=Dict(pairs(kw))
    objective=get(kwdict, :objective, "reg:squarederror")
    if (objective in validobjectives)== false
        msg="Requested objective not supported - xgboost_fit()"
        error(msg)
    end
    num_round=get(kwdict, :num_round, 0)
    if num_round<1
        msg="num_round not specified - xgboost_fit()"
        error(msg)
    end
    if testdata==[]
        msg="Test data required for model creation - xgboost_fit()"
        error(msg)
    end
    traindm=DMatrix(traindata) # train must be convertable to DMatrix
    testdm=DMatrix(testdata)  # test must be convertable to DMatrix
    hastestdata=(size(testdm)[1]>0)
    if hastestdata==false
        msg="Test data required for model creation - xgboost_fit()"
        error(msg)
    end
    if modelname!="" 
        modelname=modelname * ": "
    end

    # define train target
    yvals=XGBoost.getinfo(traindm,Float64,:label)
    nys=length(yvals)
    yidx=zeros(Int64,nys)
    if objective=="multi:softprob"
        mclasses=unique(sort(yvals))
        nclass=length(mclasses)
        ydict=Dict{Number,Int64}()
        for i in 1:nclass
            ydict[mclasses[i]]=i
        end
        for i in 1:nys
            yidx[i]=ydict[yvals[i]]
        end
    end

    #define test target
    ytest=XGBoost.getinfo(testdm,Float64,:label)
    
    # construct model
    if objective != "multi:softprob"
        mboost=xgboost(traindm,watchlist=[]; kw...) 
    else
        if classweights != [-1.0]
            if classweights==[0.0]
                classweights=ones(nclass)
                yct=countmap(yvals)
                yctmax=maximum(collect(values(yct)))
                for ks in keys(yct)
                    #yct[ks]=yctmax / yct[k2]
                    classweights[ydict[ks]]= yctmax / yct[ks]
                end
            end
            nweights=length(classweights)
            if nweights != nclass
                msg="number of class weights does not match number of classes - xgboost_model()"
                error(msg) 
            end
            ywt=Vector{Union{Int64,Float64}}(undef,nys)
            for i in 1:nys
                ywt[i]=classweights[ydict[yvals[i]]]
            end
            XGBoost.setinfo!(traindm,"weight",ywt)
        end
        mboost=xgboost(traindm,watchlist=[]; kw...) 
    end

    # construct predictions
    predtrain=XGBoost.predict(mboost,traindm)
    predtest=XGBoost.predict(mboost,testdm)
    
    # initialize return structures
    allplots=[]
    metricdata=Dict{String,Any}()

    # calculate model complexity (i.e. splits) 
    mtrees=trees(mboost)
    nsplitmodel= nsplitstree.(mtrees)

    if objective=="reg:squarederror"
        # tree diagnostics
        plt1=histogram(nsplitmodel,label="",xlabel="number of splits",
                        ylabel="number of trees", title=modelname*"splits per tree")
        push!(allplots,plt1)
        plt1= plot(collect(1:num_round),nsplitmodel, label="", xlabel="round number",
                        ylabel="number of splits", title=modelname*"splits per round")
        push!(allplots,plt1)

        # standard regression output
        ttl=modelname*"true vs prediction - test data"
        errtest= ytest .- predtest
        rmsetest=  sqrt( sum(errtest .^ 2 )/length(errtest))
        maetest= mean(abs.(errtest))
        metricdata["rmse"]=rmsetest
        metricdata["mae"]=maetest
        errmean=mean(errtest)
        errstd=std(errtest)
        errnormal= (errtest .- errmean) ./ errstd

        plt1=scatter(predtest,ytest, label="", legend=:bottomright ,
                            title=ttl, xlabel="prediction", ylabel="true value")
            
        lmdf=DataFrame(yhat=predtest,y=ytest)
        lmform=term("y")~term("yhat") 
        lmreg=lm(lmform, lmdf)
        m=coeftable(lmreg).cols[1][2]
        b=coeftable(lmreg).cols[1][1]
        p=coeftable(lmreg).cols[4][2]
        if p<.0001
            pstr=" (<0.0001)"
        else
            pstr=string(round(p,digits=4))
        end
        rsq=r2(lmreg)
        x1 , x2 =xlims(plt1)[1] , xlims(plt1)[2]
        y1 , y2 =ylims(plt1)[1] , ylims(plt1)[2]
        yrng=y2-y1
        plot!([x1,x2],[(m*x1+b),(m*x2+b)], label="regression line", linewidth=3)
        annotate!(x1,.9*yrng+y1,text("R-square = "*string(round(rsq,digits=3)), 
                        halign= :left, valign= :bottom, pointsize=10) )
        annotate!(x1,.84*yrng+y1,text("     slope = "*string(round(m,digits=3)), 
                        halign= :left, valign= :bottom, pointsize=10) )
        annotate!(x1,.78*yrng+y1,text("  p-value = "*pstr, 
                            halign= :left, valign= :bottom, pointsize=10) )
        plot!([x1,x2],[x1,x2], linewidth=2 , label="line of unity" )
        
        push!(allplots,plt1)
        metricdata["rsquared"]=rsq
        metricdata["pvalue"]=p
        metricdata["slope"]=m
        metricdata["testy"]=predtest
        metricdata["testyhat"]=ytest

        plt2= histogram(errtest, label="", xlabel="residuals",
                            title=modelname * "residual distribution - test data")
        vline!([0],label="", linewidth=2)
        push!(allplots,plt2)
        
        plt3= scatter(predtest,errnormal, label="", xlabel="prediction", 
                        ylabel="normalized residual",
                        title=modelname*"normalized residual - test data")
        plot!([xlims(plt3)[1],xlims(plt3)[2]],[0,0], label="")
        push!(allplots,plt3)

        nobs=length(errnormal)
        sort!(errnormal)
        stdNormal=Normal(0,1)
        qtheory = [quantile(stdNormal,i/(nobs+1)) for i in 1:nobs]
        plt4=scatter(qtheory,errnormal, label="", xlabel="theorteic quantile",
                        ylabel="normalized residual", 
                        title=modelname*"Q-Q plot - test data")
        plot!([qtheory[1],qtheory[end]], [qtheory[1],qtheory[end]] ,label="")
        push!(allplots,plt4)
        

    end

    if objective=="binary:logistic"
        # tree diagnostics
        plt1=histogram(nsplitmodel,label="",xlabel="number of splits",
                        ylabel="number of trees", title=modelname*"splits per tree")
        push!(allplots,plt1)
        plt1= plot(collect(1:num_round),nsplitmodel, label="", xlabel="round number",
                        ylabel="number of splits", title=modelname*"splits per round")
        push!(allplots,plt1)

        # discrimination tests
        testroc=mkroc(predtest , ytest)
        plt1=plotroc(testroc, modelname)
        push!(allplots,plt1)
        metricdata["auc"]=testroc.auc
        metricdata["threshdata"]=testroc.df
        metricdata["targetproportion"]=testroc.proportion
        plt2=plotss(testroc, modelname)
        push!(allplots,plt2)
        costcurve=plotcost(testroc, modelname, costmatrix=costmatrix)
        push!(allplots,costcurve.plot)
        metricdata["costdata"]=costcurve.costdata

        # calibration test if length(ytest)>=45
        # minimum obs per bin set to 15
        nyt=length(ytest)
        if nyt>=45
            nbin= convert(Int,floor(nyt/15))
        end
        nbin= minimum([10,nbin])
        sbin= convert(Int,round(nyt/nbin, digits=0))
        orderyt=sortperm(predtest)
        phl=predtest[orderyt]
        ythl=ytest[orderyt]
        chi=0
        xhl=Vector{Float64}(undef,nbin)
        yhl=Vector{Float64}(undef,nbin)
        for i in 1:(nbin-1)
            ohl= sum( ythl[((i-1)*sbin+1):(i * sbin)] )
            ehl= sum( phl[((i-1)*sbin+1):(i * sbin)] )
            chi = chi + (ohl-ehl)^2/ehl
            xhl[i]= ehl/sbin
            yhl[i]= ohl/sbin
        end
        ohl= sum( ythl[((nbin-1)*sbin+1):nyt] )
        ehl= sum( phl[((nbin-1)*sbin+1):nyt] )
        chi = chi + (ohl-ehl)^2/ehl
        xhl[nbin]= ehl/length(((nbin-1)*sbin+1):nyt)
        yhl[nbin]= ohl/length(((nbin-1)*sbin+1):nyt)
        chisqp=ccdf(Chisq(nbin-2),chi)
        chidata=DataFrame(observed=yhl, expected=xhl)
        if chisqp<.01
            pstr="(< .01)"
        else
            pstr=string(round(chisqp,digits=2))
        end
        if chisqp<=.05
            passtr="FAILED"
        else
            passtr="PASSED"
        end
        plt1=bar(xhl,yhl,label="", xlabel="expected", ylabel="observed",
                    title=modelname * "calibration curve - test data")
        xymax=minimum([xlims(plt1)[2],ylims(plt1)[2]])
        xymin=maximum([xlims(plt1)[1],ylims(plt1)[1]])
        plot!([xymin,xymax],[xymin,xymax], label="")
        xmin=xlims(plt1)[1]
        xrng=xlims(plt1)[2]-xmin
        ymin=ylims(plt1)[1]
        yrng=ylims(plt1)[2]-xmin
        annotate!(.05*xrng+xmin, .95*yrng+ymin,text("sample-n = "*string(length(ytest)), 
                    halign= :left, valign= :bottom, pointsize=10) )
        annotate!(.05*xrng+xmin, .90*yrng+ymin,text("Chi-sq = "*string(round(chi, digits=1)), 
                    halign= :left, valign= :bottom, pointsize=10) )
        annotate!(.05*xrng+xmin, .85*yrng+ymin,text("d.f. = "*string(nbin-2), 
                    halign= :left, valign= :bottom, pointsize=10) )
        annotate!(.05*xrng+xmin, .8*yrng+ymin,text("p value = "*pstr, 
                    halign= :left, valign= :bottom, pointsize=10) )
        annotate!(.05*xrng+xmin, .75*yrng+ymin,text("Calibration : "*passtr, 
                    halign= :left, valign= :bottom, pointsize=10) )
        push!(allplots,plt1)

        metricdata["calibrationpvalue"]=chisqp
        metricdata["calibrationchisquare"]=chi
        metricdata["calibrationchidata"]=chidata

    end

    if objective=="multi:softprob"
        if ndims(predtest)>1  # workaround for type 'transpose' returned by xgboost.jl
            predtest=Matrix(predtest)
        end
        # tree diagnostics
        nsplitmodel=reshape(nsplitmodel,(nclass,num_round))
        nsplitmodel=Matrix(transpose(nsplitmodel))
        for i in 1:nclass
            nsm=nsplitmodel[:,i]
            plt1=histogram(nsm,label="",xlabel="number of splits",
                        ylabel="number of trees", 
                        title=modelname*"[class "*string(i-1)*"] "*"splits per tree")
            push!(allplots,plt1)
            plt1= plot(collect(1:num_round),nsm, label="", xlabel="round number",
                        ylabel="number of splits", 
                        title=modelname*"[class "*string(i-1)*"] "*"splits per round")
            push!(allplots,plt1)
        end

        # create an ROC for each class
        for i in 1:nclass
            cpred=predtest[:,i]
            cy= (ytest .== (i-1))
            croc=mkroc(cpred,cy)
            plt1=plotroc(croc, modelname*"[class-"*string(i-1)*"] ")
            push!(allplots,plt1)
            metricdata["auc:class-"*string(i-1)]=croc.auc
        end
        # normalize predictions
        npred=size(predtest)[1]
        classtest=zeros(Int64,npred)
        for j2 in 1:npred
            predtest[j2,:]= predtest[j2,:] ./ sum(predtest[j2,:]) 
            classtest[j2] = findmax(predtest[j2,:])[2] - 1
        end

        # contingency table  ytest vs classtest
        crossmat=zeros(Int, nclass+1,nclass+1)
        crossnames= "pred: " .* string.(collect(0:(nclass-1)))
        tnames= "true: " .* string.(collect(0:(nclass-1)))
        push!(crossnames,"total") ;push!(tnames,"total")
        
        for i in 1:npred
            crossmat[Int(ytest[i]+1),classtest[i]+1] += 1
        end
        for i in 1:nclass
            crossmat[i,(nclass+1)]= sum(crossmat[i, 1:nclass])
            crossmat[(nclass+1),i]= sum(crossmat[1:nclass, i])
        end
        crossmat[(nclass+1),(nclass+1)]= sum(crossmat[1:nclass,1:nclass])
        xtabledf=hcat(DataFrame(crossTable=tnames),DataFrame(crossmat, crossnames))
        metricdata["contingencytable"]=xtabledf
        acclist=[]
        for i in 0:(nclass-2)
            tsum = 0
            for j in 1:nclass
                lj=j-i ; rj=j+i
                lj= (lj<1) ? 1 : lj
                rj= (rj>nclass) ? nclass : rj
                tsum += sum(crossmat[(j),lj:rj])
            end
            push!(acclist,tsum)
        end
        acclist= acclist ./ crossmat[(nclass+1),(nclass+1)]
        tnames= string.(collect(0:(nclass-2)))
        accnames= "distance: " .* tnames
        accdistdf=  DataFrame(distance=accnames,accuracy=acclist)
        metricdata["accuracybydistance"]=accdistdf
        plt1=bar(acclist , label="", xlabel="class distance", ylabel="accuracy",
                    xticks=(1:(nclass-1),tnames),
                    title=modelname * "Accuracy by distance - test data")
        ylims!(0,1.05)
        push!(allplots,plt1)

        # calculate relative cost value
        # contigency table in crossmat[1:npred, 1:npred], costs in costmatrix
        if size(costmatrix) != (nclass,nclass)
            costmatrix=ones(nclass,nclass)
            for i in 1:nclass
                costmatrix[i,i]=0
            end
        end
        costmat= (crossmat[1:nclass,1:nclass] .* costmatrix) ./ crossmat[nclass+1,nclass+1]
        maxclass=findmax(crossmat[1:nclass,nclass+1])
        defaultcost= (costmatrix[1:nclass,maxclass[2]] .* crossmat[1:nclass,nclass+1]) ./ crossmat[nclass+1,nclass+1]
        metricdata["totalcost"]=sum(costmat)
        metricdata["relativecost"]=sum(costmat)/sum(defaultcost)

        if showplots==true
            vscodedisplay(xtabledf)
        end
    end

    # report variable importance
    vidf=DataFrame(importancetable(mboost))
    variabledata=Dict{String,Any}()
    sort!(vidf,[:total_gain], rev=true)
    tgain=sum(vidf.total_gain)
    vidf.portion_total_gain= vidf.total_gain ./ tgain
    vidf.cum_gain=cumsum(vidf.portion_total_gain)
    featlist=vidf.feature
    variabledata["importance"]=vidf
    # importance control
    if importancecontrol>0.0
        if importancecontrol<1.0
            fnv=findfirst(x->x>importancecontrol, vidf.cum_gain)
            if isnothing(fnv)
                nv=length(featlist)
            else
                nv=maximum([1,(fnv-1)])
            end
        else
            nv=convert(Int64,round(importancecontrol,digits=0))
            if nv>length(featlist)
                nv=length(featlist)
            end
        end
    else
        nv=length(featlist)
    end
    # bar graph 
    plt1=bar(vidf.portion_total_gain[1:nv], orientation=:h, label="", 
                title=modelname*"variable importance",
                    yticks=(1:nv, featlist[1:nv]), yflip=true)
    push!(allplots, plt1)
    variabledata["selectedviacontrol"]=featlist[1:nv]

    # produce partial dependencies
    if typeof(traindata)== XGBData && objective!="multi:softprob"
        if (traindata.ylabel != "") && (traindata.ylabel in names(traindata.xdata))== true
            pdpdata=plotpdp(mboost, featlist[1:nv],traindata.xdata[!, Not(Symbol(traindata.ylabel))],
                      predtrain , modelname)
        else
            pdpdata=plotpdp(mboost, featlist[1:nv],traindata.xdata[!, :],
                      predtrain , modelname)
        end
        # add to allplots
        for f in featlist[1:nv]
            push!(allplots,pdpdata[f].plot)
        end
    else
        # pdp plots not well defined for multi-classification
        pdpdata=Dict{String,NamedTuple}()  # holding construction
    end

    # display plots
    np=length(allplots)
    if showplots==true  && np>0
        for i in np:-1:1
            display(allplots[i])
        end
    end

    # compose return structure  
    return ( booster=mboost, params=params, metrics=metricdata, variables=variabledata,
                     pdp=pdpdata, plots=allplots ) 

end

########  suppport functions  ##########

"""
    mkroc()

    Utility function to create threshold, sensitivity, and specificity data.  
    Result is a named tuple:
        auc => auc measurement
        df => DataFrame with data
        proportion => proportion of target

"""
function mkroc(preds,cats)
    x1=sortperm(preds)
    p1=preds[x1]
    c1=cats[x1]
    c2= 1 .- c1
    proportion=mean(c1)
    sumc1= sum(c1)
    sumc2= sum(c2)
    tpr= 1 .- (cumsum(c1) ./ sumc1)
    tpr= vcat(1.0, tpr, 0.0)
    fpr= 1 .- (cumsum(c2) ./ sumc2)
    fpr= vcat(1.0, fpr, 0.0)
    mtpr= (tpr[1:(end-1)] .+ tpr[2:end]) ./ 2
    dfpr= (fpr[1:(end-1)] .- fpr[2:end]) 
    auc =   round(sum(mtpr .* dfpr) , digits=3)
    thresh= vcat(0.0, p1, 1.0)
    df=DataFrame( thresh= thresh , tpr=tpr, fpr=fpr)

    return (auc=auc, df=df, proportion=proportion)

end

"""
    plotroc()

    Utility function to plot ROC

"""
function plotroc(rc, modelname)

    pltroc=plot(rc.df.fpr, rc.df.tpr,linewidth=3, label="", title= modelname*"ROC curve - test data")
    xlims!(0,1)
    ylims!(0,1.05)
    plot!([0,1],[0,1], label="", color= :black)
    annotate!(.84,.5,text("AUC = "*string(round(rc.auc,digits=3)), 
                        halign= :right, valign= :bottom, pointsize=12) )

    return pltroc
end

"""
    plotss()

    Utility function to plot sensitivity vs specificity

"""
function plotss(rc, modelname )
    pltss=plot(rc.df.thresh, rc.df.tpr,linewidth=3, label="Sens." , legend=:bottomright ,
                            xlabel="Threshold" , ylabel="Sensitivity / Specificity",
                            title= modelname*"threshold metrics - test data")
    xlims!(0,1.05)
    ylims!(0,1.05)
    spec= 1 .- rc.df.fpr
    #acc=(rc.proportion .* rc.df.tpr) .+ ((1.0 .- rc.proportion) .* spec )

    plot!(rc.df.thresh, spec,linewidth=3, label="Spec.")
    #plot!(rc.df.thresh, acc,linewidth=3, label="Acc.")
    opt=findmin(abs.(rc.df.tpr .- spec))
    #vline!(pltss, [rc.df.thresh[opt[2]]], label="", c=:black )
    plot!([rc.df.thresh[opt[2]],rc.df.thresh[opt[2]]],[0,spec[opt[2]]], label="",
                            linewidth=1, color= :black)
    annotate!(rc.df.thresh[opt[2]]+ .02, .1,text(string(round(rc.df.thresh[opt[2]],digits=3)),
                        halign= :left, valign= :bottom, pointsize =8))
    plot!([rc.df.thresh[opt[2]]+ .05,rc.df.thresh[opt[2]]],
                [.09,0], arrow=true, linewidth=1, 
                   color= :black, label="")
    plot!([0,rc.df.thresh[opt[2]]],[spec[opt[2]],spec[opt[2]]],linewidth=1,
                    color= :black, label="")
    annotate!(0.02,spec[opt[2]],text(string(round(spec[opt[2]],digits=3)),
                    halign= :left, valign= :bottom, pointsize =8))

    return pltss
end

"""
    plotcost()

    Utility function to plot cost vs threshold

"""
function plotcost(rc, modelname ; costmatrix=[1])
    # cost curve for binary classes
    if size(costmatrix)!= (2,2)
        costmatrix= [0.0 1.0 ; 1.0 0.0]
    end
    ctn=costmatrix[1,1] ; ctp=costmatrix[2,2]
    cfn=costmatrix[2,1]  ; cfp=costmatrix[1,2]
    
    tnr= 1 .- rc.df.fpr
    fnr= 1 .- rc.df.tpr
    cstabs=  (ctp .* rc.df.tpr ) .+ (cfp .* rc.df.fpr ) .+ (ctn .* tnr ) .+ (cfn .* fnr )
    cr0= cstabs[end]
    cr= cstabs ./ cr0

    pltcost=plot(rc.df.thresh, cr,linewidth=3, label="rel. cost" , legend=:bottomright ,
                            xlabel="Threshold" , ylabel="Relative Cost", minorgrid=true ,
                            gridalpha=0.7 , minorgridalpha=0.3,
                            title= modelname*"cost metrics - test data")
    ylims!(0,ylims(pltcost)[2])
    y1=0
    yrng=ylims(pltcost)[2]
    xlims!(0,xlims(pltcost)[2])
    x1= 0
    xrng=xlims(pltcost)[2]
    annotate!(.95*xrng+x1,.3*yrng+y1,text("Cost FP: "* string(round(cfp,digits=2)),
                    halign= :right, valign= :bottom, pointsize =8 , color= :red) )
    annotate!(.95*xrng+x1,.25*yrng+y1,text("Cost FN: "* string(round(cfn,digits=2)),
                    halign= :right, valign= :bottom, pointsize =8 , color= :red) )
    annotate!(.95*xrng+x1,.2*yrng+y1,text("Cost TP: "* string(round(ctp,digits=2)),
                    halign= :right, valign= :bottom, pointsize =8 , color= :red) )
    annotate!(.95*xrng+x1,.15*yrng+y1,text("Cost TN: "* string(round(ctp,digits=2)),
                    halign= :right, valign= :bottom, pointsize =8 , color= :red) )

    costdata=DataFrame(treshold=rc.df.thresh, relativecost=cr, absolutecost=cstabs)
    return (plot=pltcost , costdata=costdata)
end

"""
    plotpdp()

    Utility function to calculate and plot partial dependence plots

"""
function plotpdp(bst::Booster,featlist::Vector{String}, data::DataFrame ,yhat, modelname)
    
    pdpdata=Dict{String,NamedTuple}()
    meanyhat=mean(yhat)
    nfeat=length(featlist)
    ypmax=-Inf
    ypmin=Inf

    for i in 1: nfeat

        feat=featlist[i]
        featdata=data[!,Symbol(feat)]
        featvals=unique(sort(featdata))
        featvals=collect(skipmissing(featvals))
        featnum=length(featvals)
        if featnum>24
            featscan=quantile(featvals,range(start=.025,stop=.975, length=25), sorted=true)
        else
            featscan=featvals
        end
        nscan=length(featscan)
        featspan=maximum(featscan)-minimum(featscan)
        ypdp=Vector{Float64}(undef,0)
        ndata=size(data)[1]
        

        for j in 1:nscan
            scandf= copy(data)
            scandf[!,Symbol(feat)]= repeat([featscan[j]],ndata)
            scanpred=XGBoost.predict(bst,scandf)
            push!(ypdp,mean(scanpred))
        end
        #global max/min
        ypmax2=maximum(ypdp)
        ypmax=maximum([ypmax,ypmax2])
        ypmin2=minimum(ypdp)
        ypmin=minimum([ypmin,ypmin2])

        # calculate partial dependence of missing value
        scandf= copy(data)
        scandf[!,Symbol(feat)]= repeat([missing],ndata)
        scanpred=XGBoost.predict(bst,scandf)
        misspdp=mean(scanpred)

        if nscan>10
            plotpdp=plot(featscan,ypdp, label="" , title=modelname*"pdp - "*feat,
                           ylabel="mean partial prediction", xlabel=feat, linewidth=2)
            hline!(plotpdp , [meanyhat], label="mean\ntraining\nprediction" , legend=:outerbottomright )
            model = loess(featscan, ypdp, span=0.5)
            us = range(extrema(featscan)...; step = 0.25 * featspan/(nscan-1))
            vs = Loess.predict(model, us)
            plot!(us,vs, label="", linewidth=3)
            scatter!([xlims(plotpdp)[1]],[misspdp],markershape=:star5,label="missing\nvalue")

        else
            svals=string.(featvals)
            plotpdp=bar(ypdp, xticks=(1:featnum, svals),label="" , title=modelname*"pdp - "*feat,
                            ylabel="mean\npartial\nprediction", xlabel=feat, bar_width=1/featnum)
            hline!(plotpdp , [meanyhat], label="mean\ntraining\nprediction" , legend=:outerbottomright)
            scatter!([xlims(plotpdp)[1]],[misspdp],markershape=:star5,label="missing\nvalue")

        end
        
        pdpdata[feat]= ( plot=plotpdp, x=featscan, y=ypdp, misspdp=misspdp, meanyhat=meanyhat)
    end 

    # standardize y-axis
    for key in keys(pdpdata)
        v=pdpdata[key]
        plt=v.plot
        ylims!(plt,(0.95*ypmin,1.05*ypmax))
        if v.misspdp>ypmax || v.misspdp<ypmin
            annotate!(plt,xlims(plt)[1],ypmax, text("  missing = " * string(round(v.misspdp,sigdigits=3)) ,
                                halign= :left, valign= :top, pointsize =8 , color= :red))
        end
        pdpdata[key]= ( plot=plt, x=v.x, y=v.y, misspdp=v.misspdp, meanyhat=v.meanyhat)
    end

    return (pdpdata)
end

"""
    traveltree()

    Utility function that recursively travels down an xgboost tree

"""
function traveltree(tree::XGBoost.Node, leaflevel::Vector{Int64})
    kids=tree.children
    if !isempty(kids)
        for kid in kids
            if length(kid.children)==0
                push!(leaflevel,1)
            end
            traveltree(kid, leaflevel)
        end
    end
end
"""
    nsplitstree()

    Utility function that calculates number of splits per xgboost tree
        
"""
function nsplitstree(tree::XGBoost.Node)
    leaves=[0]
    traveltree(tree,leaves)
    splits=maximum([0,sum(leaves)-1])
    return (splits)
end