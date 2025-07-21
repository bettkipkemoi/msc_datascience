import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

# Function for linear fit
def linear(x, a, b):
    #TODO
    return a * x + b

# Function for polynomial fit (2nd degree as example)
def polynomial(x, a, b, c):
    #TODO
    return a * x**2 + b * x + c

# Function for exponential fit
def exponential(x, a, b, c):
    #TODO
    return a * np.exp(b * x) + c

# Generate synthetic data
def generate_data(func, params, x_range):
    x = np.linspace(*x_range, 100)
    y = func(x, *params)
    noise = np.random.normal(0, 0.1, size=y.shape)
    y_noisy = y + noise
    return x, y_noisy

# Fit the data to a given model
def fit_curve(x, y, model_type):
    if model_type == 'linear':
        #TODO
        popt, _ = curve_fit(linear, x, y)
        fitted_curve = linear(x, *popt)
    elif model_type == 'polynomial':
        #TODO
        popt, _ = curve_fit(polynomial, x, y)
        fitted_curve = polynomial(x, *popt)
    elif model_type == 'exponential':
        #TODO
        popt, _ = curve_fit(exponential, x, y)
        fitted_curve = exponential(x, *popt)
    else:
        raise ValueError("Unsupported model type. Choose 'linear', 'polynomial', or 'exponential'.")
    
    return popt, fitted_curve

# Analyze residuals
def analyze_residuals(y, fitted_curve):
    #TODO
    residuals = y - fitted_curve
    rmse = np.sqrt(np.mean(residuals**2))
    return residuals, rmse

# Calculate goodness of fit (R-squared)
def calculate_r_squared(y, fitted_curve):
    #TODO
    ss_residual = np.sum((y - fitted_curve) ** 2)
    ss_total = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_residual / ss_total)
    return r_squared

# Main function to run the tool
def fit_and_analyze(x_range=(0, 10), model_type='linear'):
    # Step 1: Generate synthetic experimental data
    x, y = generate_data(linear, (2, 5), x_range)  # You can change this line to generate different data

    # Step 2: Fit the model to the data
    #TODO
    popt, fitted_curve = fit_curve(x, y, model_type)

    # Step 3: Analyze residuals and goodness of fit
    #TODO
    residuals, rmse = analyze_residuals(y, fitted_curve)
    r_squared = calculate_r_squared(y, fitted_curve)

    # Step 4: Visualize results
    plt.scatter(x, y, color='blue', label='Experimental Data')
    plt.plot(x, fitted_curve, color='red', label=f'Fitted {model_type.capitalize()} Curve')
    plt.title(f'Curve Fitting: {model_type.capitalize()}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

    # Output results
    print(f"Fitted Parameters: {popt}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R-squared: {r_squared}")

# Example usage
if __name__ == "__main__":
    fit_and_analyze(model_type='linear')
    fit_and_analyze(model_type='polynomial')
    fit_and_analyze(model_type='exponential')
