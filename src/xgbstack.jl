# functions to combine multiple boosters and assess stability in Results


function timesec()
    n=now()
    hour(n)*3600+minute(n)*60+second(n)
end