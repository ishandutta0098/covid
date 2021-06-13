import fbprophet
from fbprophet import Prophet

model_fb = Prophet(
            growth = 'linear', 
            seasonality_mode = 'multiplicative',  
            changepoint_prior_scale = 30,
            seasonality_prior_scale = 15,
        )