import numpy as np
import pandas as pd
from fbprophet import Prophet

def faz_predicao(X, tempo_em_dias):
    model = Prophet()
    model.fit(X)
    future = model.make_future_dataframe(periods=tempo_em_dias, include_history=True)
    future.tail()
    forecast = model.predict(future)
    forecast[['ds','yhat', 'yhat_lower', 'yhat_upper']].tail()
    RMSE = np.sqrt(np.mean(forecast.loc[:X.shape[0],'yhat'] - X['y']) ** 2)
    forecast['RMSE'] = pd.Series(RMSE, index=forecast.index)
    print('RMSE: %f' % RMSE)
    model.plot(forecast,uncertainty=True)
    model.plot_components(forecast)
    return forecast

def main():
    df = pd.read_csv('international-airline-passengers.csv')
    ds = pd.DataFrame()
    ds['ds'] = df['Month']
    ds['y'] = df['Passengers']
    previsao = faz_predicao(ds,10)
    print(previsao.columns)
    print(previsao)

if __name__ == '__main__':
    main()

