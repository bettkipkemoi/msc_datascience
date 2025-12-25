library(forecast)
library(tseries)

# 1. Load the data
# Make sure you've downloaded 'DailyTotalFemaleBirths.csv' from Kaggle and saved in working directory.
data <- read.csv("DailyTotalFemaleBirth.csv")

# Inspect the data structure
str(data)
head(data)

# The dataset typically has columns: 'Date' and 'Births'.
# Convert the 'Date' column to Date class
data$Date <- as.Date(data$Date, format="%Y-%m-%d")

# Create a time series object
# Since these are daily births for 1959, frequency = 365 is often used for daily data (non-leap year)
births_ts <- ts(data$Births, start=c(1959,1), frequency=365)

# 2. Test for stationarity
# We can use the Augmented Dickey-Fuller (ADF) test:
adf_result <- adf.test(births_ts)
adf_result

# Interpretation:
# If the p-value is small (<0.05), we can reject the null hypothesis of non-stationarity.
# In this dataset, the daily female births data is often already stationary or nearly so.
# If not stationary, we would consider differencing:
# births_ts_diff <- diff(births_ts)

# (Check stationarity on differenced series if needed)
# adf.test(births_ts_diff)

# 3. Split the data into training and testing sets for model evaluation
# Let's hold out the last 30 days as test data.
train_length <- length(births_ts) - 30
train_ts <- window(births_ts, end=c(1959,(train_length/365)*365))
test_ts <- window(births_ts, start=c(1959, ((train_length+1)/365)*365))

# 4. Fit an ARIMA model using auto.arima on the training set
fit <- auto.arima(train_ts)
summary(fit)

# Check the residuals of the fitted model
checkresiduals(fit)

# If residuals look like white noise (no patterns, no autocorrelations), the model is adequate.

# 5. Forecast on the test period
# Forecast for the length of the test set (30 days)
fcast <- forecast(fit, h=30)
plot(fcast)
lines(test_ts, col="red", type="l")  # Add the actual test data in red

# Evaluate forecasting accuracy
accuracy_metrics <- accuracy(fcast, test_ts)
accuracy_metrics

# Look at RMSE, MAE, MAPE, etc. to assess forecast accuracy.

# 6. If satisfied with the model, you can refit using the entire dataset and forecast future values
final_fit <- auto.arima(births_ts)
fcast_future <- forecast(final_fit, h=30) # forecasting next 30 days beyond 1959
plot(fcast_future)