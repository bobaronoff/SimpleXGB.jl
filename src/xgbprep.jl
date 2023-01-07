# base routines for auto xgb

using DataFrames
#using SimpleDFutils

using Statistics, StatsBase, Random

"""
    xgboost_prep()

    This function is designed to prepare a DataFrame for training with XGBoost.

    Two un-named parameters are for the dataframe and column name of the target variable.

    When the target is nominal or ordinal, the classes to be kept are listed in
    'targetstrings' (i.e. Vector{String}). If not specified, all unique values in the
    column are used (max number is 6). If the first listed is 'other', the function
    will group all values not listed in remainder of 'targetstrings'.

    Two named parameters, 'includex' and 'excludex' are Vector{String} which indicate
    columns to be used in the model.  In general, only one of these two would be used.

    'parsetonum'=true instructs to attempt conversion of {String} columns to numeric.

    {String} columns with 10 or less unique values will be treated as nominal and
    encoded to multiple columns using either one-hot or dummy method as specified by
    'dummyoveronehot' (default=false). The dummy method removes the column with least frequency.

    Nominal or ordinal targets require a minimum number of each class as indicated
    in parameter 'mintargetnum'. The parameter 'targetasordinal'=true instructs
    to treat a nominal target as ordinal.  The order is as indicated in 'targetstrings'.
    If 'targetstrings' is empty, the order is alphabetic.

    The function accommodates 'missing' values in the DataFrame. When a target column has a 
    missing value, the entire row is removed. A 'missing' value in a variable column is left
    in place.  A variable column is removed if the fraction 'missing' exceed 'maxmissingfrac'
    (default is 0.9).

    This function separates the DataFrame into training and test sets.  The fraction alotted 
    to the training portion is 'trainfraction' (default is 0.8). Nominal and ordinal targets
    are divided by class.

    The function returns a named tuple.
        prepdf => prepared DataFrame
        regclass => regression class (i.e. numeric, logistic, multinomial,ordinal)
        targetstrings => class names of target
        xnamedict => a dictionary with class names of co-variates
        trainindex => vector of row index for training set
        testindex => vector or row index for test set
        map => named tuple ( recipes => recipe map, postnames => naming map)
        map=(recipes=recipes, postnames=postnames)
    
        Example
        ```
        preppedtuple=xgboost_prep(mydf,"y",excludex=["ID"], mintargetnum=40)
        anotherprep=boostpref(mydf2,"y", targetstrings=["mild","medium","severe"],
                                includex=["x1","x2","z1"], 
                                targetasordinal=true, trainfraction=.7)
        ```

"""
function xgboost_prep( predf::DataFrame, targetname::String; 
                        targetstrings::Vector{String}=Vector{String}(undef,0),
                        includex::Vector{String}=Vector{String}(undef,0), 
                        excludex::Vector{String}=Vector{String}(undef,0), 
                        parsetonum::Bool=true,dummyoveronehot::Bool=false ,
                        mintargetnum::Int64=50, targetasordinal::Bool=false , 
                        trainfraction::Float64=0.8 , maxmissingfrac::Float64=0.9 )

    # make copy to make proper without altering original
    df=copy(predf)
    properdf!(df)
    tallydf=typetally(df)
    grpother=0
    if length(targetstrings)>0
        if lowercase(targetstrings[1])=="other"
            grpother=1
        end
    end
    
    # locate target and determine Julia Type
    tidx=findfirst(isequal.(targetname),tallydf.ColNames)
    if isnothing(tidx)
        msg="targetname not in DataFrame : xgboost_prep() ."
        error(msg)
    end
    vtypecounts=tallydf[tidx,2:end]
    vtypenames=names(tallydf)[2:end]
    t2idx=findbyrowdf([0,0],vtypecounts,equal=false)
    if length(t2idx)==0
        msg="Target is not showing any Type : xgboost_prep() ."
        error(msg)
    end
    ttypenames=vtypenames[t2idx]
    ntypes=length(findall(.!isequal.("Missing",ttypenames)))
    if ntypes>1
        msg="Target is showing multiple Types : xgboost_prep() ."
        error(msg)
    end
    # process Target type
    ttype=findfirst(.!isequal.("Missing",ttypenames)) 
    typeallowed=["String","Float64","Bool","Int64","Int32","Int16"]
    typetarget=ttypenames[ttype]
    if !(typetarget in typeallowed)
        msg="Target Type: " * typetarget * " not in list of included Type - xgboost_prep() ."
        error(msg)
    end
    targetcounts=vtypecounts[t2idx[ttype]]
    ttype=findfirst(isequal.("Missing",ttypenames))
    if isnothing(ttype)
        targetmissing=[0,0]
    else
        targetmissing=vtypecounts[t2idx[ttype]]
    end

    if trainfraction<0.5
        trainfraction=0.5
    end
    if trainfraction>0.9
        trainfraction=0.9
    end
    trainindex=Vector{Int64}(undef,0)
    testindex=Vector{Int64}(undef,0)

    # determine X-variable list
    xnames=copy(tallydf[:,1])
    if length(excludex)>0
        xnames=setdiff(xnames,excludex)
    end
    if length(includex)>0
        xnames=intersect(xnames,includex)
    end
    xnames=setdiff(xnames,[targetname])
    nxnames=length(xnames)
    if nxnames==0
        msg="Criteria does not leave any 'X' variables. \n" 
        msg=msg * "Check 'includex' and 'excludex' parameters. - xgboost_prep() .\n"
        error(msg)
    end

    # targetname [as string], targettype [as string],targetcounts [total, unique], targetmissing [counts,]
    # tidx is the column in df with targetcounts
    boostdf=DataFrame()
    allowmissing!(boostdf)
    regclasses=["numeric","logistic","multinomial","ordinal"]
    regclass="unassigned"
    # determine the category of regression i.e. continuous, binary(i.e. logistic), multinomial, ordinal
    tunique=targetcounts[2]
    tmissing=targetmissing[1]
    # is target number or string? If string, is targetstrings correct?
    if typetarget=="String"
        if tunique<2
            msg="Nominal target requires at least 2 unique values - xgboost_prep() ."
            error(msg)
        end
        if tunique>5
            msg="Nominal target exceeds maximum unique values (i.e. 5) - xgboost_prep() ."
            error(msg)
        end
        if tunique==2
            regclass="logistic"
        else
            regclass="multinomial"
            if targetasordinal==true
                regclass="ordinal"
            end
        end
        if length(targetstrings)==0
            targetstrings=unique(sort(df[!,tidx]))
            filter!(x->!ismissing(x),targetstrings)
        end
        tdict=Dict{String,Int64}()
        nt2=length(targetstrings)
        for i in 1:nt2
            tdict[targetstrings[i]]= (i-1)
        end
        lvy=length(df[!,tidx])
        vy=Vector{Union{Int64,Missing}}(undef,lvy)
        for i in 1:lvy
            if grpother==0
                vy[i]=get(tdict,df[i,tidx],missing)
            else
                if ismissing(df[i,tidx])
                    vy[i]=missing
                else
                    vy[i]=get(tdict,df[i,tidx],0)
                end
            end
        end
        vycount=countmap(vy)
        vymissing=get(vycount,missing,0)
        vypresent=lvy-vymissing
        if vypresent<mintargetnum
                msg="Mapping of the target found less than minimum " * string(mintargetnum) * " targets.\n"
                msg= msg * "Check that 'targetstrings' parameter is correct."
                error(msg)
        end
        vyfreq=zeros(nt2)
        msg=""
        for i in 1:nt2
            vynum=get(vycount,(i-1),0)
            vyfreq=vynum/vypresent
            if vyfreq*vypresent<mintargetnum
                msg=msg * "Number of target '" * targetstrings[i] * "' (" * string(vynum) * 
                            ") less than minimum " * string(mintargetnum) * " targets.\n"
                #msg=msg * "Number of target '" * targetstrings[i] * "' less than minimum " * mintargetnum
            end
            if vyfreq>.9
                msg=msg * "Frequency of target '" * targetstrings[i] * "' > 0.9\n"
            end
        end
        if msg != ""
            msg="Target imbalance - xgboost_prep()\n" * msg
            error(msg)
        end
        
        boostdf[!,Symbol(targetname)]=vy
  
    end
    if typetarget in ["Float64","Int64","Int32","Int16"]
        
        regclass="numeric"
        vy=copy(df[!,tidx])
        lvy=length(vy)      
        uvy=unique(sort(vy))
        filter!(x->!ismissing(x),uvy)
        nvy=length(uvy)
        if nvy<6
            if targetasordinal==true || uvy==[0,1] || uvy==[-1,1]
                if nvy==2
                    regclass="logistic"
                else
                    regclass="ordinal"
                end
                tdict=Dict{Any,Int64}()
                nt2=length(uvy)
                for i in 1:nt2
                    tdict[uvy[i]]= (i-1)
                end
                ty1=[eltype(vy) , Int64, Missing]
                vy=convert(AbstractVector{Union{ty1...}},vy)
                for i in 1:lvy
                    vy[i]=get(tdict,vy[i],missing)
                end
                vy=convert(AbstractVector{Union{Int64,Missing}},vy)
            else
                msg="There are 5 or less numeric target values. \nThese can only be processesed with"
                msg=msg * "'targetasordinal=true' - xgboost_prep()"
                error(msg)
            end
        end
        #lvy=length(vy)
        vymissing=count(ismissing.(vy))
        vypresent=lvy-vymissing
        if vypresent<mintargetnum
                msg="Mapping of the target found less than minimum " * mintargetnum * " targets."
                error(msg)
        end
        boostdf[!,Symbol(targetname)]=vy
    end

    if typetarget=="Bool"
        regclass="logistic"
        vy=copy(df[!,tidx])
        if tmissing==0
            vy=convert(AbstractVector{Int64},vy)
        else
            vy=convert(AbstractVector{Union{Int64,Missing}},vy)
        end
        lvy=length(vy)
        vycount=countmap(vy)
        vymissing=get(vycount,missing,0)
        vypresent=lvy-vymissing
        if vypresent<mintargetnum
                msg="Mapping of the target found less than minimum " * mintargetnum * " targets - xgboost_prep() ."
                error(msg)
        end
        tvar=mean(skipmissing(vy))
        if tvar<.1 || tvar>.9
            msg="Binary Target proportion out of accepted range [0.1-0.9] - xgboost_prep() ."
            error(msg)
        end
        boostdf[!,Symbol(targetname)]=vy
    end
    
    # build variable data and convert nominal in to hot-encoding
    # xnames => list of X variables ; tallydf => dataframe of type tallies
    vtypenames=names(tallydf)[2:end]
    xnamedict=Dict{String,Vector{String}}()
    postnames=Dict{String,Vector{String}}()
    recipes=Dict{String,String}()

    for i in 1:nxnames
        xidx=findfirst(isequal.(xnames[i]),tallydf.ColNames)
        if isnothing(xidx)
            msg="variable '" * xnames[i] *"' not in DataFrame - xgboost_prep() ."
            error(msg)
        end
        vtypecounts=tallydf[xidx,2:end]
        x2idx=findbyrowdf([0,0],vtypecounts,equal=false)
        if length(x2idx)==0
            msg="Variable '" * xnames[i] * "' is not showing any Type - xgboost_prep() ."
            error(msg)
        end
        xtypenames=vtypenames[x2idx]
        xtypes=findall(.!isequal.("Missing",xtypenames))
        xmtypes=findall(isequal.("Missing",xtypenames))
        if length(xtypes)>1
            msg="Variable '" * xnames[i] * "' is showing multiple Types - xgboost_prep() ."
            error(msg)
        end
        if length(xtypes)==0
            msg="Variable '" * xnames[i] * "' is not showing any non-missing Type - xgboost_prep() ."
            error(msg)
        end
        xtype=xtypenames[xtypes[1]]
        xtally1= vtypecounts[x2idx[xtypes]][1][1]
        xtally2= vtypecounts[x2idx[xtypes]][1][2]
        if length(xmtypes)>0
            xmissing=vtypecounts[x2idx[xmtypes]][1][1]
        else
            xmissing=0
        end
        if xtally1==xtally2 && xtype=="Int64"
            msg="Variable '" * xnames[i] * "' appears to be an Indentity field - xgboost_prep() ."
            error(msg)
        end
        if xtally2==1
            msg="Variable '" * xnames[i] * "': has no variation - xgboost_prep() ."
            error(msg)
        end
        if xmissing/(xmissing+xtally1)> maxmissingfrac
            msg="Variable '" * xnames[i] * "': fraction missing exceeds max("
            msg= msg * string(maxmissingfrac) * ") - xgboost_prep() ."
            error(msg) 
        end

        vx=df[!,xidx]
        recipes[xnames[i]]="asis"
        
        if xtally2>10 && xtype=="String"
            if parsetonum==true
                # convert string to number
                vx=numparse(vx)
                recipes[xnames[i]]="parsenum"
                rtype=unique(typeof.(vx))
                xtype=string(filter!(x->x!=Missing, rtype)[1])
            end
            if xtype=="String"
                recipes[xnames[i]]="pass"
                msg="Variable '" * xnames[i] * "': nominal type exceeds max(10) - xgboost_prep() ."
                error(msg)
            else
                xtally2=length(findall(ismissing.(vx)))
                xtally1=length(unique(vx))
                if xtally2>0
                    xtally1 = xtally1-1
                end
            end
        end
        if xtype=="Bool"
            lvx=length(vx)
            newvx=Vector{Union{Missing,Int64}}(missing,lvx)
            bvx=findall(.!ismissing.(newvx))
            newvx[bvx]=convert.(Int64,skipmissing(vx))
            vx=newvx
            xtype="Int64"
            recipes[xnames[i]]="bool2int"
        end
        # xnames[i], xtype (as string), xtally1 (total entry) , xtally2 (unique entry),xmissing (num missing)
        # add vx to boostdf
        if xtype !="String"
            boostdf[!,Symbol(xnames[i])]=vx
            xnamedict[xnames[i]]=[xnames[i]]
            postnames[xnames[i]]=[xnames[i]]
        else
            # area to hot-encode
            # ux = unique(df.x); transform(df, @. :x => ByRow(isequal(ux)) .=> Symbol(:x_, ux))
            ux=unique(vx)
            hasmissing= ( count(isequal.(missing,ux))>0 )
            sx=xnames[i]
            tempdf=DataFrame()
            tempdf[!,Symbol(sx)]=vx
            recipes[xnames[i]]="onehot"
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
            # ?remove column with least number for hot-encoding
            if dummyoveronehot==true
                ncols=size(tempdf)[2]
                ctone=zeros(Int64,ncols)
                for c in 1:ncols
                    ctone[c]=sum(skipmissing(tempdf[!,c]))
                end
                cminidx=findmin(ctone)[2]
                namemin=names(tempdf)[cminidx]
                tempdf=tempdf[!,Not(Symbol(namemin))]
                recipes[xnames[i]]="dummy"
            end
            # add temdf to boostdf
            boostdf=hcat(boostdf,tempdf)
            xnamedict[xnames[i]]=newxnames
            postnames[xnames[i]]=newxnames
        end

    end
    
    # remove rows where target is Missing
    tmiss2=findall(ismissing.(boostdf[!,Symbol(targetname)]))
    delete!(boostdf,tmiss2)

    # divide into training/test set
    
    if regclass=="unassigned"
        msg="unable to assign a supported regression type to dataset - xgboost_prep() ."
        error(msg)
    end
    if regclass=="numeric"
        # split numeric target
        ndfrows=size(boostdf)[1]
        ntrain= convert(Int64,round(trainfraction*ndfrows, digits=0))
        # ntest=ndfrows-ntrain
        tempidx=collect(1:ndfrows)
        shuffle!(tempidx)
        append!(trainindex,tempidx[1:ntrain])
        append!(testindex,tempidx[ntrain+1:ndfrows])
        sort!(trainindex)
        sort!(testindex)
    else
        # split target by individual target values
        vy=boostdf[!,Symbol(targetname)]
        vylist=unique(sort(vy))
        filter!(x->!ismissing(x),vylist)
        nvylist=length(vylist)
        for i in 1:nvylist
            tempidx=findall(isequal.(vylist[i],vy))
            ndfrows=length(tempidx)
            ntrain= convert(Int64,round(trainfraction*ndfrows, digits=0))
            shuffle!(tempidx)
            append!(trainindex,tempidx[1:ntrain])
            append!(testindex,tempidx[ntrain+1:ndfrows])
        end
        sort!(trainindex)
        sort!(testindex)
    end



    # return named tuple
    return (prepdf=boostdf, regclass=regclass, targetstrings=targetstrings , xnamedict=xnamedict,
                 trainindex=trainindex, testindex=testindex, 
                    map=(recipes=recipes, postnames=postnames))
end

function reversedict(xdict::Dict{String, Vector{String}})
    newdict=Dict{String,String}()
    for (k,v) in xdict
        for x in v
            newdict[x]=k
        end
    end
    return newdict
end


"""
    findbyrowdf()

    This function scans a DataFrameRow and returns index(es) that match parameter 'findme'.
    The 'equal' parameter determines if matches for equal(i.e. true) or not-equal(i.e. false)[default=> 'equal = true']

    Example
    ```
    result= findbyrowdf(somevalue,mydataframerow)
    ```

"""
function findbyrowdf(findme::Any,rowdf::DataFrameRow ; equal::Bool=true)
    fidx=Vector{Int64}(undef,0)
    nr=length(rowdf)
    if nr>0
        for i in 1:nr
            if equal && rowdf[i] == findme
                push!(fidx,i)
            end
            if !equal && rowdf[i] != findme
                push!(fidx,i)
            end
        end
    end
    return(fidx)
end

"""
    numparse()

    This function parses a Vector{String} into either Vector{Int64} or Vector{Float64} dependent on the values within 'str'.  
    This function accommodates missing values.  
    Individual values that can not be parsed are converted to value 'missing'.  
    If the result percent missing exceeds 50%, the original Vector{String} is returned.

    Example```
    mynumbers=numparse(mystringvector)
    ```
    
"""
function numparse(str::Vector{Union{Missing, String}})
    #str .= ifelse.(isnothing.(str), missing, str)
    ls=length(str)
    m1=findall(.!ismissing.(str))
    str2=str[m1]
    ls2=length(str2)
    p1=tryparse.(Int64,str2)
    np1=count(isnothing.(p1))
    p2=tryparse.(Float64,str2)
    np2=count(isnothing.(p2))
    if np2<np1
        # Float64
        if np2>0.5 * ls2
                # too many failed parsing to allow conversion
                return str
        end
        nstr=Vector{Union{Float64,Missing}}(missing,ls2)
        nstr2=Vector{Union{Float64,Missing}}(missing,ls)
        g1=findall(.!isnothing.(p2))
        nstr[g1]=p2[g1]
        nstr2[m1]=nstr
        hasmissing= ( count(isequal.(missing,nstr2))>0 )
        if hasmissing==false
            nstr2=convert(AbstractVector{Float64}, nstr2)
        end
        return nstr2
    else
        # Int64
        if np1>0.5 * ls2
            # too many failed parsing to allow conversion
            return str
        end
        nstr=Vector{Union{Int64,Missing}}(missing,ls2)
        nstr2=Vector{Union{Int64,Missing}}(missing,ls)
        g1=findall(.!isnothing.(p2))
        nstr[g1]=p2[g1]
        nstr2[m1]=nstr
        hasmissing= ( count(isequal.(missing,nstr2))>0 )
        if hasmissing==false
            nstr2=convert(AbstractVector{Int64}, nstr2)
        end
        return nstr2
    end
end

function numparse(str::Vector{String})
    str=convert(AbstractVector{Union{Missing, String}}, str)
    numparse(str)
end

"""
    hotstuff()

    This function converts a Vector{Union{Missing, String}} into a DataFrame.
    The columns represent either One-Hot or Dummy encoding (dependent on 'dummyoveronehot').
    One-hot via dummyoveronehot=false (default).

    'xname' will be the rootname of the encoded variables.

    'missing' values are distributed to each column unless keepmissingcol=true.
    This creates a separate column to demarcate 'missing' status.
    
    Example
    ```
    codedDF=hotstuff(mystringvector,"X", dummyoveronehot=true)
    ```
    
"""
function hotstuff(vx::Vector{Union{Missing, String}}, xname::String ; keepmissingcol::Bool=false , dummyoveronehot::Bool=false)
    if xname==""
        msg="xname parameter not specified - hotstuff() ."
        error(msg)
    end
    ux=unique(vx)
    hasmissing= ( count(isequal.(missing,ux))>0 )
    sx=xname
    tempdf=DataFrame()
    tempdf[!,Symbol(sx)]=vx
    transform!(tempdf, @. Symbol(sx) => ByRow(isequal(ux)) .=> Symbol(sx *"_", ux))
    tempdf=tempdf[!,Not(Symbol(sx))]
    mapcols!(col -> convert.(Int64, col), tempdf)
    if hasmissing==true
        if keepmissingcol==false
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
    end
    # ?convert one-hot to dummy encoding
    if dummyoveronehot==true
        ncols=size(tempdf)[2]
        ctone=zeros(Int64,ncols)
        for c in 1:ncols
            ctone[c]=sum(skipmissing(tempdf[!,c]))
        end
        cminidx=findmin(ctone)[2]
        namemin=names(tempdf)[cminidx]
        tempdf=tempdf[!,Not(Symbol(namemin))]
    end
        
    return tempdf
end

function hotstuff(vx::Vector{String} , xname::String ; keepmissingcol::Bool=false , dummyoveronehot::Bool=false)
    vx=convert(AbstractVector{Union{Missing, String}}, vx)
    hotstuff(vx, xname, keepmissingcol=keepmissingcol, dummyoveronehot=dummyoveronehot)
end