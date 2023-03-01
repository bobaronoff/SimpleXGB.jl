# functions to combine multiple boosters and assess stability in Results
function xgboost_stack(traindata; 
    testdata::Any=[],
    nstack::Int=10,
    #modelname::String="",
    #showplots::Bool=true,
    #costmatrix=[1.0],
    #classweights=[-1.0],
    #importancecontrol::Union{Float64,Int64}=-1,
    kw... )



end



function timesec()
    n=now()
    hour(n)*3600+minute(n)*60+second(n)
end