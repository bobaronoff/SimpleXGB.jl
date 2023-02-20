# routines for xgboost_shap


"""

    This function bypases XGBoost.predict in order to obtain Shapley values.

    The 'type' parameter conforms to prediction types specified in the XGBoost documentation.
    Options include:
        0 => normal (default)
        1 => output margin
        2 => predict contribution
        3 => predict approximate contribution
        4 => predict feature interactions
        5 => predict approximate feature interactions
        6 => predict leaf training (see XGBoost documentation)
    The shape of returned data varies with 'type' option and certain objectives.
    
"""
function predict_shapley(b::Booster, Xy::DMatrix;
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


"""
    xgboost_shap()

    This function performs and plots SHAP analysis.  
    Shapley values obtained with TreeSHAP algorithm.
    Returned are plots for SHAP importance, dependency, and contribution.

    First parameter is a booster object.  
    The second parameter is data to be scored and passed to SHAP analysis.
        The form of data must be convertable to DMatrix and compatible with 
        booster object.

    There are several optional named parameters:
        estimate => {Bool} to indicate whether to approximate Shapley values
                    (default = false)
        topnvar => {Int} number of variables to plot (default=0 -> all variables)
        modelname => {String} name on plots 
        spanloess => {Float64} span[0.0,1.0] for loess line on dependency plots  
                     (default = 0.0 -> no line)
        showplots => {Bool} triggers display of plots (default=true)
        stndardizeplots => {Bool} places all dependency plots on same y axis limits
                            ( default= false)
        shapcolor => any prefined color gradient or user defines gradients accepted
                     by Plots.jl (default = :rainbow)
        shapalpha => {Float64} any alpha value [0.0,1.0] accepted by Plots.jl (default= 0.5)
    
    There are four objects returned in a named tuple.
        plots => a vector of plots
        bias => the bias value for contributions. This is generally the average prediction
                of the supplied data (on the object margin scale).
        shapley => the shapley values corresponding to the scored data. returned as Matrix
                   type to allow further analysis.  Column names correspond to the 
                   'feature_names' field of the supplied booster object.
        importance => a dataframe with feature names and Shapley value based importance.
                      Importance is the mean of absolute value of Shapley values for a feature.
                      Rows are ordered from most to least important.

    Example:
    ```
        myshap= xgboost_shap(my_booser, train_data, topnvar=8);
    ```

"""
function xgboost_shap(b::Booster, Xy;
    estimate::Bool=false,
    topnvar::Integer=0, # 0 => all variables
    modelname::String = "" ,
    spanloess::Float64=0.0,  # 0 => no loess line
    showplots::Bool=true ,
    standardizeplots::Bool=false,
    shapcolor= :rainbow ,
    shapalpha= 0.5   
   )
    # function to provide SHAP analyses
    featlist=b.feature_names
    objective=get(b.params,:objective,"reg:squarederror")
    if objective[1:5]=="multi"
        msg="SHAP analysis not yet defined for multi-class objectives."
        error(msg)
    end
    type= estimate ? 3 : 2
    
    dmXy=DMatrix(Xy)
    if typeof(Xy)==XGBData
        Xy=Xy.xdata[!,Not(Symbol(Xy.ylabel))]
    end

    if modelname != ""
        modelname= modelname * " : "
    end

    allplots=[]

    shap_data=predict_shapley(b,dmXy,type=type)
    # mean_contrib=mean(shap_data[:,1:(end-1)])
    # std_contrib=std(shap_data[:,1:(end-1)])
    #quant_contrib=quantile(vec(shap_data[:,1:(end-1)]),[.025,.975])
    shaplohi=extrema(vec(shap_data[:,1:(end-1)]))

    shap_imp=  mapslices(mean,abs.(shap_data), dims=1)
    bias=shap_imp[end]
    shap_imp=shap_imp[1:(end-1)]
    imporder=sortperm(shap_imp, rev=true)
    if topnvar<1 || topnvar>length(shap_imp)
        nfeat=length(shap_imp)
    else 
        nfeat=topnvar
    end
    varimp=DataFrame(feature=featlist[imporder], importance=shap_imp[imporder])

    shap_c=abs.(shap_data[:,(1:(end-1))])
    t_shap=mapslices(sum,shap_c, dims=2)
    nc=size(shap_data)[2]-1
    X_p=zeros(size(Xy))
    X_p=convert(AbstractMatrix{Union{Missing, Float64}},X_p)
    for i in 1:nc
        shap_c[:, i]= shap_c[:, i] ./ t_shap
        #= mxmn=quantile(skipmissing(Xy[:,i]),[.005,.995])
        frng=mxmn[2]-mxmn[1]
        X_p[:,i]=  (Xy[:,i] .- mxmn[1]) ./ frng
        X_p[:,i]=map(x-> ismissing(x) ? missing : x<0.0 ? 0.0 : x ,X_p[:,i])
        X_p[:,i]=map(x-> ismissing(x) ? missing : x>1.0 ? 1.0 : x ,X_p[:,i]) =#
        X_p[:,i]= percentilerangevalue(Xy[:,i])
    end
    shap_c_x=extrema(shap_c)
    
    # plot variable importance via Shapley values
    plt1=bar(shap_imp[imporder[1:nfeat]], orientation=:h, label="", 
                title=modelname * "Shapley based variable importance", color=shapcolor,
                xlabel="mean(absolute(Shapley Value))",
                yticks=(1:nfeat, featlist[imporder[1:nfeat]]), yflip=true)
    
    
    push!(allplots,plt1)

    if showplots==true
        display(plt1)
    end

    gX=Vector{Float64}(undef,0)
    gY=Vector{Float64}(undef,0)
    gZ=Vector{Float64}(undef,0)
    gZ2=Vector{Float64}(undef,0)

    # create SHAP dependence plot
    for i in 1:nfeat
        #plt1=plotshapdp(Xy[!:Symbol(featlist[imporder[i]])],shap_data[!,imporder[i]])
        xnotmiss=findall(!ismissing,Xy[:, Symbol(featlist[imporder[i]])])
        sX=Xy[xnotmiss, Symbol(featlist[imporder[i]])]
        sY=shap_data[xnotmiss,imporder[i]]
        sZ=shap_c[xnotmiss,imporder[i]]
        sZ2=X_p[xnotmiss,imporder[i]]
        jY=jit(sY)
        fY=fill(Float64(nfeat-i+1), length(sY)) .+ jY
        append!(gX,sY)
        append!(gY,fY)
        append!(gZ,sZ)
        append!(gZ2,sZ2)
        xlohi=quantile(sX,[.005,.995])
        plt1=scatter(sX,sY, xlim=(xlohi[1],xlohi[2]),
                     title=modelname * "Shapley dependence - " * featlist[imporder[i]], 
                     markersize=3, label="", marker_z=sZ, color=shapcolor,clims=shap_c_x,
                     colorbar_title="fractional contribution to prediction",
                     ylabel="margin contribution")
        if standardizeplots==true
            plot!(plt1, ylim=shaplohi)
        end
        if length(unique(sort(Xy[xnotmiss, Symbol(featlist[imporder[i]])])))>10  && spanloess  >0.0
            # plot loess line
            sX=convert(Vector{Float64},sX)
            model=loess(sX,sY,span=spanloess)
            xrng=range(extrema(sX)...;length=25)
            yloess=Loess.predict(model,xrng)
            plot!(xrng,yloess, linewidth=3, label="")
        end
        
        push!(allplots,plt1)

        if showplots==true
            display(plt1)
        end
    end

    #full TreeShap plot contribution
    plt1=scatter(gX,gY, label="", title=modelname * "TreeSHAP contributions", ylim=(0.5,0.5+nfeat),
                    yticks=(1:nfeat, featlist[imporder[nfeat:-1:1]]), marker_z=gZ,
                    markersize=2,color=shapcolor, markeralpha=shapalpha, 
                    colorbar_title="fractional contribution to prediction",
                    xlabel="Shapley value")
    vline!([0.0],label="")
    
    push!(allplots,plt1)    

    if showplots==true
        display(plt1)
    end
    
    #full TreeShap plot summary
    plt1=scatter(gX,gY, label="", title=modelname * "TreeSHAP summary", ylim=(0.5,0.5+nfeat),
                    yticks=(1:nfeat, featlist[imporder[nfeat:-1:1]]), marker_z=gZ2,
                    markersize=2,color=shapcolor, markeralpha=shapalpha, 
                    colorbar_title="feature value (percentile range)",
                    xlabel="Shapley value")
    vline!([0.0],label="")
    
    push!(allplots,plt1)    

    if showplots==true
        display(plt1)
    end

    return (plots=allplots, bias=bias, shapley=shap_data,importance=varimp )
end

function jit(x)
    dx=mapdensity(x)
    maxd=maximum(dx)
    ndx= dx ./ maxd
    r=(rand(length(x)) .- 0.5) .* 0.8
    jx= ndx .* r
    return jx
end


function mapdensity(x)
    u=kde(x)
    d=Vector{Float64}(undef,0)
    for i in eachindex(x)
        push!(d,u.density[findmin(abs.(u.x.-x[i]))[2]])
    end
    return d
end

function percentilerangevalue(x)
    idx=findall(!ismissing,x)
    x2=convert(AbstractVector{Float64},x[idx])
    u=kde(x2)
    cmdf=cumsum(u.density)
    cmdf= cmdf ./ cmdf[end]
    cd=Vector{Float64}(undef,0)
    for i in eachindex(x2)
        push!(cd,cmdf[findmin(abs.(u.x.-x2[i]))[2]])
    end
    cdx=extrema(cd)
    cdrng=cdx[2]-cdx[1]
    cd= (cd .- cdx[1]) ./ cdrng
    x3=Vector{Union{Missing,Float64}}(missing,length(x))
    x3[idx]=cd
    return x3
end