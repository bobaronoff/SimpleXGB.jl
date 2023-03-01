# functions to combine multiple boosters and assess stability in Results

#returns vector of booster objects
function xgboost_stack(traindata; nstack::Int=10,verbose=true,kw... )

    #strip out 'seed' (if exists) from keyword tuple
    kwdict=Dict(pairs(kw))
    delete!(kwdict,"seed")
    delete!(kwdict,"watchlist")
    kw=(;kwdict...)

    traindm=DMatrix(traindata) # train must be convertable to DMatrix

    boosters=Vector{XGBoost.Booster}(undef,0)
    for s in 1:nstack
        if verbose==true
            println("Start model: "*string(s)*" - @["*Dates.format(now(), "HH:MM::SS")*"]")
        end
        push!(boosters,xgboost(traindm,watchlist=[],seed=timesec(); kw...) )
    end
    return boosters

end

function xgboost_stack_predict(boosters::Vector{XGBoost.Booster},pdata)

    pdm=DMatrix(pdata)
    nstack=length(boosters)
    # predictions done on margin
    pr=predict_shapley(boosters[1],pdm,type=1)
    if nstack>1
        for i in 2:nstack
            pr2=predict_shapley(boosters[i],pdm,type=1)
            pr= pr .+ pr2
        end
        pr = pr ./ nstack
    end
    # convert margins in to output (based on objective)
    objective=get(boosters[1].params, :objective, "reg:squarederror")
    if objective in ["binary:logistic","multi:softprob"]
        pr= map(logit2prob,pr)
    end
    
    return pr
end



function timesec()
    n=now()
    hour(n)*3600+minute(n)*60+second(n)
end

function logit2prob(logit)
  odds = exp(logit)
  prob = odds / (1 + odds)
  return(prob)
end