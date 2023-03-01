module SimpleXGB

# Utility functions to create XGBoost models

using DataFrames,Dates
#using SimpleDFutils
using XGBoost
using Statistics, StatsBase, Random
using Distributions , GLM , Loess
using Plots
using JSON3
using KernelDensity
using Dates

# structures
export XGBData

# functions
export xgboost_prep, xgboost_lc , xgboost_set
export xgboost_fit, xgboost_score
export xgboost_log, nsplitstree
export xgboost_shap 

include("xgbprep.jl")

include("xgblearn.jl")

include("xgbfit.jl")

include("xgbscore.jl")

include("xgbshap.jl")





include("xgbdata.jl")  # this needs to be last entry

end
