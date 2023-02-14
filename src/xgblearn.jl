using DataFrames
using XGBoost
using Random
using Plots
using Dates

"""
    xgboost_lc()

    This function performs n-fold cross validation in the creation of an XGBOOST model.
    The number of rounds are specified with the named keyord 'num_round'.  The number of
    folds designated by 'nfolds' parameter.  The tracking metric is the xgboost default 
    alternate(s) supplied to the string vector 'evals'.

    The function presents two plots. Presented is a comparison of the average of training set
    vs fold metrics.  Multiple graphs are presented for multiple metrics.  Metric results 
    come from xgboost routines. Some of the non-default metrics may not be stable.  Also 
    presented are the number of average splits per tree vs rounds. This provides feedback
    on model complexity and effects of regularization parameters.

    All routine XGBOOST parameters are entered as keyword parameters.

    Control parameters 'verbose' and 'showplot' affect behavior.

    Results are returned as a named tuple:
        metrics => a dataframe returning train/test metric data
        splitdata => a matrix of the tree split data
        plots => a vector of the displayed plots
        params => a dictionary with xgboost parameters

    Example:

    ```
    myxgb=xgboost_lc(mydata, objective="binary:logisitc", subsample=.5, max_depth=5, num_round=100, eta=.01)
    
    myxgb2=xgboost_lc(mydata2,objective="reg:squarederror", max_depth=6,num_round=500, 
    eta=.01, subsample=.05, evals=["rmse","mae"], modelname="myModel")

    ```


"""
function xgboost_lc(data;
                    num_round::Int64=100,
                    nfolds::Int64=10,
                    evals::Vector{String}=Vector{String}(undef,0),
                    showplots::Bool=true,
                    modelname::String="",
                    verbose::Bool=true,
                    kw...)

    # function to create xgboost learning curve(s) using n-fold cross validation
    
    # initialize
    datadm=DMatrix(data)  # 'data' must be convertable to DMatrix
    numrows=size(datadm)[1]
    foldidx= shuffle!((1:numrows) .% nfolds)
    neval=maximum([1,length(evals)])
    metrics=zeros(num_round,(2*neval))
    ttrain=0 ; ttest=0
    rtnevalnames= [""]
    params= Dict()
    nr=0

    if verbose==true
        println("Starting "*string(nfolds)*"-fold CV")
        println("  N= "*string(numrows)*" @["*Dates.format(now(), "HH:MM::SS")*"]")
    end

    # execute folds and tally metrics
    for i in 0:(nfolds-1)
        trainidx=findall(.!isequal.(i,foldidx))
        testidx=findall(isequal.(i,foldidx))
        ftrain=length(trainidx) ; ftest=length(testidx)
        ttrain += ftrain ; ttest += ftest
        foldrun=xgboost_log(datadm[trainidx, :], testdata=datadm[testidx,:], num_round=num_round, evals=evals;kw...)
        ftrees=trees(foldrun.booster)
        nr=size(foldrun.log)[1]
        nsplitfold= nsplitstree.(ftrees)
        if i==0
            rtnevalnames=names(foldrun.log)
            params=foldrun.booster.params
            nsplitall=zeros(length(nsplitfold))
        end
        nsplitall .+= nsplitfold
        for j in 2:(1+neval)
            metrics[:,(j-1)] .+= (ftrain .* foldrun.log[!,j])
            metrics[:,(neval+j-1)] .+= (ftest .* foldrun.log[!,neval+j])
        end
        if verbose==true 
            println("Complete fold-"*string(i+1)*" @["*Dates.format(now(), "HH:MM::SS")*"]");
        end
    end
    if verbose==true
        println("Computing results"*" @["*Dates.format(now(), "HH:MM::SS")*"]");
    end
    for i in 1:neval
        metrics[:,i] ./= ttrain
        metrics[ : , i+neval] ./=ttest
    end
    nsplitall = nsplitall ./ nfolds

    # create learning curves
    if modelname != "" ; modelname=modelname * ": " ; end
    allplots=[]
    splitcalc=mksplitplot(nsplitall,nr, modelname,params)
    push!(allplots,splitcalc.plot)
    for i in 1:neval
        nmeval= split(rtnevalnames[i+1],"-")[2]
        push!(allplots,mklcplot(metrics[:,i],metrics[:,i+neval], nmeval, modelname,params))
    end
    
    if showplots==true
        for i in eachindex(allplots)
            display(allplots[i])
        end
    end
    if verbose==true
        println("Learning curve complete"*" @["*Dates.format(now(), "HH:MM::SS")*"]");
    end
    # build return
    metricdf=DataFrame(metrics,rtnevalnames[2:end])
    return(metrics=metricdf, splitdata=splitcalc.splitdata, plots=allplots , params=params)

end

"""
    mksplitplot()

    This is a utility function that creates the tree split plots.  
        Not meant to be called directly.

"""
function mksplitplot(nsplitall,numrounds, modelname,params)
    ns= Int(length(nsplitall)/numrounds)
    if ns>1
        nsplitall=reshape(nsplitall,(ns,numrounds))
        nsplitall=Matrix(transpose(nsplitall))
    end
    if ns==1
        nsplitall=movave(nsplitall,7)
        plt1= plot(collect(1:numrounds),nsplitall, label="", xlabel="round number",
                    ylabel="ave. number of splits", linewidth=2,
                    title=modelname*"ave. splits each round")
    else
        ls=reshape(string.(collect(0:(ns-1))),(1,ns))
        for i in 1:size(nsplitall)[2]
            nsplitall[:,i]=movave(nsplitall[:,i],7)
        end
        plt1= plot(collect(1:numrounds),nsplitall, label=ls, xlabel="round number",
                    ylabel="ave. number of splits", linewidth=2,legend_title="Class",
                    legend_title_font_pointsize=8, legend_title_font_halign=:left,
                    title=modelname*"ave. splits each round")
    end
    xmarg= xlims(plt1)[1] + .035*(xlims(plt1)[2]-xlims(plt1)[1])
    ylims!(plt1,(0,ylims(plt1)[2]))
    plist=string.(collect(keys(params))) ; pval=string.(values(params)) ; pc=0
    for i in eachindex(plist)
        if plist[i]!="eval_metric"
            pc += 1
            annotate!(xmarg, ylims(plt1)[1]+(.05*pc)*(ylims(plt1)[2]-ylims(plt1)[1]),
            text(plist[i]*": "*pval[i], 
            halign= :left, valign= :bottom, pointsize=8))
        end    
    end
    return (plot=plt1 , splitdata=nsplitall)
end

"""
    mklcplot()

    This is a utility function that creates the learning curve plot(s).  
        Not meant to be called directly.
        
"""
function mklcplot(train, test, nmeval,ctitle, params)
    if ctitle==""
        ttl="XGBoost: CV Learning Curve"
    else
        ttl=ctitle * "CV Learning Curve"
    end
    roundsteps=collect(1:length(train))
    best= (test[end]<test[1]) ? findmin(test) : findmax(test)
    pltbest= (test[end]<test[1]) ? 1.1 : 0.9  # used in plotting
    
    plt1=plot(roundsteps,[train,test],linewidth=3,
                    title=ttl, label=["train" "test"],
                    xlabel="rounds", ylabel=nmeval,
                    legend=:bottomleft)
    # show optimal rounds
    vline!(plt1, [best[2]], label="", c=:black )
    # show origin
    ylims!(plt1,(0,ylims(plt1)[2]))
    # arrow to optimal rounds
    plot!(plt1,[.92*best[2],best[2]],[pltbest*best[1],best[1]], 
                    arrow=true, color=:black,linewidth=1 , label="")
    annotate!(best[2], .2*ylims(plt1)[2],text(string(best[2]), 
                    halign= :right, valign= :bottom, pointsize=8, rotation = 90))
    annotate!(.92*best[2], pltbest*best[1],text(string(round(best[1], sigdigits=3)), 
                    halign= :right, valign= :bottom, pointsize=8))
    # show parameters
    xmarg= xlims(plt1)[1] + .035*(xlims(plt1)[2]-xlims(plt1)[1])
    plist=string.(collect(keys(params))) ; pval=string.(values(params)) ; pc=0
    for i in eachindex(plist)
        if plist[i]!="eval_metric"
            pc += 1
            annotate!(xmarg, (0.12+.05*pc)*ylims(plt1)[2],text(plist[i]*": "*pval[i], 
                    halign= :left, valign= :bottom, pointsize=8))
        end    
    end
    return plt1                
end

"""
    xgboost_log()

    This function will grow an XGBOOST model and capture the evaluation log.
    Treaining data is the first unnamed parameter. Any object that can implicitly 
    be converted to DMatrix form is acceptable. Testing data is passed via the named 
    parametr 'testdata'.  The length of model is specified by 'num_round".  Evaluation
    metrics are either XGBOOST defaults or specified in the vector 'evals'.

    Desired xgboost parameters need provided as keyword arguments.

    Results are returned as a named tuple.
        booster => xgboost booster
        log => a DataFrame with evaluation metrics

    Example:

    ```
    mymodel=xgboost_log(myDM, num_round=500, objective="reg:squarederror", eta=.01, 
    subsample=.6,gamma=10, max_depth=5, evals=["rmse","mae"])
    ```

"""
function xgboost_log(traindata, a...;
    testdata::Any=[] ,
    num_round::Integer=10,
    evals::Vector{String}=Vector{String}(undef,0),
    kw...
)

    Xy = XGBoost.DMatrix(traindata)  # traindata must be convertable to DMatrix
    b = XGBoost.Booster(Xy; kw...)
    b.feature_names = XGBoost.getfeaturenames(Xy)
    for i in eachindex(evals)
        XGBoost.setparam!(b,"eval_metric",evals[i])
    end
    if testdata != []
        testdm=DMatrix(testdata)  # testdata (if submitted) must be convertable to DMatrix
        names=["train","test"]
        watch=[Xy.handle, testdm.handle]
    else
        names=["train"]
        watch=[Xy.handle]
    end
    thelog = Vector{String}(undef,0)
    o = Ref{Ptr{Int8}}()
    for j in 1:num_round
        XGBoost.xgbcall(XGBoost.XGBoosterUpdateOneIter, b.handle, j, Xy.handle)
        XGBoost.xgbcall(XGBoost.XGBoosterEvalOneIter, b.handle, j, watch, names, length(watch), o)
        push!(thelog,unsafe_string(o[]))
    end
    return (booster=b , log=parsethelog(thelog))
end

"""
    parsethelog()

    This is a utility function that parses the captured xgboost evaluation log.

"""
function parsethelog(thelog::Vector{String})
    nr=length(thelog)
    neval= length(findall(":",thelog[1]))
    cstr=split(replace(replace(thelog[1],"\t"=>","),":"=>","),",")
    evalnames=cstr[collect(2:2:(2*neval))]
    vals=zeros(nr,neval)
    rnd=zeros(Int,nr)
    for r in 1:nr
        l3=split(replace(replace(thelog[r],"\t"=>","),":"=>","),",")
        rnd[r]= parse(Int64,SubString(l3[1],2,length(l3[1])-1))
        for c in 1:neval
            vals[r,c]= parse(Float64,l3[1+2*c])
        end
    end
    valdf=hcat(DataFrame(iteration=rnd),DataFrame(vals, evalnames))
    return valdf
end

"""
    movave()

    Utility function to create a moving average.
    The 'span' on each side is passed in second parameter.
    ( Total span is 2*buff+1 )

"""
function movave(X::Vector,buff::Int)
    len = length(X)
    X=vcat(fill(X[1],buff),X,fill(X[end],buff))
    Y = zeros(len)
    for n = 1:len
        Y[n] = mean(X[n:(n+2*buff)])
    end
    return Y
end

function xgboost_shapley(b::Booster, Xy::DMatrix;
                         type::Integer=0,  # 0-normal, 1-margin, 2-contrib, 3-est. contrib,4-interact,5-est. interact, 6-leaf
                         training::Bool=false,
                         ntree_lower_limit::Integer=0,
                         ntree_limit::Integer=0,  # 0 corresponds to no limit
                        )
    opts = Dict("type"=>type,
                "iteration_begin"=>ntree_lower_limit,
                "iteration_end"=>ntree_limit,
                "strict_shape"=>false,
                "training"=>training,
                ) |> JSON3.write
    oshape = Ref{Ptr{UInt64}}()
    odim = Ref{UInt64}()
    o = Ref{Ptr{Cfloat}}()
    XGBoost.xgbcall(XGBoost.XGBoosterPredictFromDMatrix, b.handle, Xy.handle, opts, oshape, odim, o)
    dims = reverse(unsafe_wrap(Array, oshape[], odim[]))
    o = unsafe_wrap(Array, o[], tuple(dims...))
    length(dims) > 1 ? permutedims(o, reverse(1:ndims(o))) : o
end