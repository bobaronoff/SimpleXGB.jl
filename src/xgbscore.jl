"""
    xgboost_score()

    This function will use a booster to score a data within a DataFrame.
    Use presumes a data preparation map produced by xgboost_prep(). 

    Map is provided via keyword argument 'map'. Extra columns within data can be
    matched to scores as a pass through designated by 'passthrux'. This is helpful
    with indentity columns or any column not used in booster but still of interest.

    The scores are returned as a DataFrame.

    Example

    ```
    myscores=xgboost_score(mybooster,mynewdatadf,map=mymap,passthrux=["ID"])

    ```


"""
function xgboost_score(booster::Booster, data::DataFrame=DataFrame(); 
                        mapping::NamedTuple,passthrux::Vector{String}=[""])
# routine to score DataFrame with Booster, DataFrame,and Recipe
    if passthrux !=[""]
        ptdf=data[!,Symbol.(passthrux)]
        xdf=data[!,Not(Symbol.(passthrux))]
    else
        xdf=data
    end
    sdf=mkrecipe(xdf,mapping)
    snames=names(sdf)
    bnames=booster.feature_names
    xtranames=setdiff(snames,bnames)
    missnames=setdiff(bnames,snames)
    if length(xtranames)>0
        sdf=sdf[!,Not(Symbol.(xtranames))]
    end
    if length(missnames)>0
        msg="Missing features: " * join(missnames,",") 
        error(msg)
    end
    # reorder sdf columns to match order of bnames
    sdf=select!(sdf, Symbol.(bnames))
    pred=XGBoost.predict(booster,sdf)
    npred=length(size(pred))
    #scdf=DataFrame(pred, :auto)
    if npred>1
        scnames="score_class" .* string.(0:(npred-1))
        scdf=DataFrame(pred, Symbol.(scnames))
    else
        scdf=DataFrame(scores=pred)
    end
    #rename!(scdf,scnames)
    if passthrux !=[""]
        scdf=hcat(ptdf,scdf)
    end
    return scdf
end

function mkrecipe(xdf::DataFrame, xmap::NamedTuple)
    #routine to transform DataFrame via recipes
    df=copy(xdf)
    xgbproperdf!(df)
    numrows=size(df)[1]
    rnames=keys(xmap.recipes)
    xnames=names(df)
    newdf=DataFrame()
    mnames=setdiff(rnames,xnames)
    if length(mnames)>0
        msg="Missing Variable(s): " * join(mnames,",") * " not  in dataframe - xgbscore()"
        error(msg)
    end
    for rname in rnames
        vx=df[!,Symbol(rname)]
        goal=xmap.recipes[rname]
        newnames=xmap.postnames[rname]
        if goal=="asis"
            # check for non-converted string
            rtype=unique(typeof.(vx))
            xtype=string(filter!(x->x!=Missing, rtype)[1])
            if xtype=="String"
                # convert string to number
                vx=numparse(vx)
                rtype=unique(typeof.(vx))
                xtype=string(filter!(x->x!=Missing, rtype)[1])
                if xtype=="String"
                    msg="Variable: " *rname* " failed to parse to numbers - xgbscore()"
                    error(msg)
                end
            end
            newdf[!,Symbol(newnames[1])]=vx
        end
        if goal=="parsenum"
            # convert string to number
            vx=numparse(vx)
            rtype=unique(typeof.(vx))
            xtype=string(filter!(x->x!=Missing, rtype)[1])
            if xtype=="String"
                msg="Variable: " *rname* " failed to parse to numbers - xgbscore()"
                error(msg)
            end
            newdf[!,Symbol(newnames[1])]=vx
        end   
        if goal=="bool2int"
            lvx=length(vx)
            newvx=Vector{Union{Missing,Int64}}(missing,lvx)
            bvx=findall(.!ismissing.(newvx))
            newvx[bvx]=convert.(Int64,skipmissing(vx))
            vx=newvx
            newdf[!,Symbol(newnames[1])]=vx
        end      
        if goal=="onehot" || goal=="dummy"
            # area to hot-encode
            # ux = unique(df.x); transform(df, @. :x => ByRow(isequal(ux)) .=> Symbol(:x_, ux))
            ux=unique(vx)
            hasmissing= ( count(isequal.(missing,ux))>0 )
            sx=rname
            tempdf=DataFrame()
            tempdf[!,Symbol(sx)]=vx
            transform!(tempdf, @. Symbol(sx) => ByRow(isequal(ux)) .=> Symbol(sx *"_", ux))
            tempdf=tempdf[!,Not(Symbol(sx))]
            mapcols!(col -> convert.(Int64, col), tempdf)
            if hasmissing==true
                tempdf=tempdf[!,Not(Symbol(sx*"_missing"))]
                # map missing from vx
                mx=findall(ismissing.(vx))
                lmx=length(mx)
                allmx=Vector{Missing}(missing,lmx)
                ncols=size(tempdf)[2]
                for c in 1:ncols
                    tempdf[!,c]=convert(AbstractVector{Union{Int64,Missing}}, tempdf[!,c]) 
                    tempdf[mx,c].= allmx
                end
            end
            newxnames=names(tempdf)
            for tname in newnames
                xidx=findfirst(isequal.(tname),newxnames)
                if isnothing(xidx)
                    println("Encoding variable: " * tname * "not found.\nMissing column inserted.")
                    vx=repeat([missing],numrows)
                    vx=convert(AbstractVector{Union{Int64,Missing}},vx) 
                    newdf[!,Symbol(tname)]=vx
                else
                    newdf[!,Symbol(tname)]=tempdf[!,xidx]
                end
            end
        end
    end
    return newdf
end