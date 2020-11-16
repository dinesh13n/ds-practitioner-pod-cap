import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from fbprophet import Prophet

def get_stationarity(df_data,rol_day):
    #rolling statistics
    rolmean = df_data.rolling(rol_day).mean()
    rolstd = df_data.rolling(rol_day).std()
    
    #plot rolling statistics
    #df_data_orig = df_data.set_index('invoice_date')
    orig = plt.plot(df_data, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #perform Dickey-Fuller test
    print('Result for Dickey-Fuller Test:')
    dftest = adfuller(df_data,autolag='AIC')
    dfoutput = pd.Series(dftest[0:4],index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical value(%s)'%key]=value
    print(dfoutput)

def log_shift(data,sft_val):
    ts_log = np.log(data)
    ts_log = ts_log.dropna()    
    ts_log_diff = ts_log - ts_log.shift(sft_val)
    ts_log_diff = ts_log_diff.dropna()
    ts_log_diff = pd.Series(ts_log_diff.values, index = ts_log_diff.index)
    return ts_log, ts_log_diff
    
def fit_ARIMA(ts,sft=1):
    '''
    Fits ARIMA model after log shifting it by unit
    Args: ts-timeseries data
          order-(p,d,q) values
          shift - unit values to shift
    Output: fitted model alongwith a plot of fitted values and RSS
    '''
    ts_log,ts_log_diff = log_shift(ts,sft)
    model = ARIMA(ts_log,order=(1,1,1))
    results_ARIMA = model.fit(disp=-1)
    plt.plot()
    plt.plot(results_ARIMA.fittedvalues,color='red')
    plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))
    return results_ARIMA

def get_continuous(df,freq):
    '''
    Makes the data continuos to be compatible with the seasonal_decompose
    function
    '''
    df = df[['price','invoice_date']].set_index('invoice_date')
    df = df.asfreq(freq=freq)
    df['price'].interpolate(inplace = True)
    return df

def get_decompose(data,col,sft):
    ts_log,ts_log_diff = log_shift(data[col],sft)
    decomposition = seasonal_decompose(ts_log,period = int(len(ts_log)/2))
    
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
    figure(num=None, figsize=(10,7), dpi=300, facecolor='w', edgecolor='k')
    plt.subplot(411)
    plt.plot(ts_log, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal,label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    
    return trend, seasonal, residual

def set_transform(data,window):
    ''' Transform the data and remove the trend and make it stationary. '''
    ts_log = np.log(data)
    avg_log = ts_log.rolling(window=window).mean()
    diff_ts_avg = (ts_log-avg_log).dropna()
    return diff_ts_avg

def get_ACF_PACF(ts,window):
    figure(num=None, figsize=(5, 3), dpi=300, facecolor='w', edgecolor='k')
    # Use the transform function data
    ts_log_diff = set_transform(ts,window=window)
    lag_acf = acf(ts_log_diff, nlags=10)
    lag_pacf = pacf(ts_log_diff, nlags=10, method='ols')
    
    #Plot ACF: 
    plt.subplot(121) 
    plt.plot(lag_acf)
    plt.axhline(y=0,linestyle='--',color='green')
    plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='red')
    plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='red')
    plt.title('Autocorrelation Function')
    
    #Plot PACF:
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0,linestyle='--',color='green')
    plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='red')
    plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='red')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()
    
def predict_ARIMA(ts,sft):
    '''
    Returns the predicted values from ARIMA model
    '''
    results_ARIMA = fit_ARIMA(ts, sft)
    predictions_ARIMA = pd.Series(results_ARIMA.fittedvalues, copy=True)
    return predictions_ARIMA

def prophet_forecast(data,period,freq,changepoint_prior_scale=0.5):
    
    data = get_continuous(data,freq)
    df = pd.DataFrame()
    df['ds'] = data.index
    df['y'] = data.price.values
    m = Prophet(changepoint_prior_scale=changepoint_prior_scale)
    m_fit = m.fit(df)
    future = m.make_future_dataframe(periods = period)
    forecast = m.predict(future)
    forecast = forecast.round(0)
    fig1 = m.plot(forecast)
    fig2 = m.plot_components(forecast)
    
    return forecast
    