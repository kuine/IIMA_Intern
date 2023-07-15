from datetime import date
import matplotlib.pyplot as plt
import numpy as np
import math
from nsepy import get_history
import pandas as pd
from datetime import date, datetime, timedelta
from scipy.stats import norm
import scipy.stats
import os

def VaR(c_name,s_date,e_date,lambda_val,confidence):
    name=c_name
    start_date = s_date
    end_date = e_date
    sdate = datetime.strptime(start_date, '%Y-%m-%d')
    edate = datetime.strptime(end_date, '%Y-%m-%d')
    data=get_history(symbol=name,start=sdate,end=edate)
    df=data
    df=df[["Close"]]
    df["returns"]=df.Close.pct_change()
    df=df.dropna()
    
    df1=df
    
    # using the exponentially weighted moving average 
    parameter = lambda_val
    std_dev=np.std(df['returns'])
    Variance = std_dev**2
    for k in range(1,df1['returns'].count()+1):
        val=((df1.iat[k-1,1])**2)
        Variance =Variance*(parameter) + (1-parameter)*val
    std_devtn1 = Variance**0.5
    mean_dailyvar=0
    c= confidence
    var = abs(norm.ppf(1-(c/100),mean_dailyvar,std_devtn1))

    # variance covariance method
    plt.clf()
    df.sort_values('returns',inplace=True,ascending=True)
    mean = np.mean(df['returns'])
    std_dev=np.std(df['returns'])
    df['returns'].hist(bins=40,density=True,histtype='stepfilled',alpha=0.5)
    x=np.linspace(mean-3*std_dev,mean+3*std_dev,100)
    plt.plot(x,scipy.stats.norm.pdf(x,mean,std_dev),"r")
    plt.savefig("templates/figure.jpg")
    var = abs(norm.ppf(1-(c/100),mean,std_dev))
    db=df[df.lt(-var, axis=1)].mean()
    output=[var,db.loc['returns']]
    return output
