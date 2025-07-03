import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np

def draw_plot():
    # Read data from file
    df = pd.read_csv('epa-sea-level.csv')
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scatter plot of the original data
    ax.scatter(df['Year'], df['CSIRO Adjusted Sea Level'], 
              color='blue', alpha=0.6, s=20, label='Data Points')
    
    # Create first line of best fit using all data
    slope_all, intercept_all, r_value_all, p_value_all, std_err_all = linregress(
        df['Year'], df['CSIRO Adjusted Sea Level']
    )
    
    # Create x values from first year to 2050
    years_extended = np.arange(df['Year'].min(), 2051)
    
    # Calculate y values for the line of best fit (all data)
    line_all = slope_all * years_extended + intercept_all
    
    # Plot the first line of best fit
    ax.plot(years_extended, line_all, 'red', linewidth=2, 
            label=f'Best fit line (all data): y = {slope_all:.4f}x + {intercept_all:.2f}')
    
    # Create second line of best fit using data from 2000 onwards
    df_recent = df[df['Year'] >= 2000]
    
    slope_recent, intercept_recent, r_value_recent, p_value_recent, std_err_recent = linregress(
        df_recent['Year'], df_recent['CSIRO Adjusted Sea Level']
    )
    
    # Create x values from 2000 to 2050
    years_recent = np.arange(2000, 2051)
    
    # Calculate y values for the recent line of best fit
    line_recent = slope_recent * years_recent + intercept_recent
    
    # Plot the second line of best fit
    ax.plot(years_recent, line_recent, 'green', linewidth=2, 
            label=f'Best fit line (2000-present): y = {slope_recent:.4f}x + {intercept_recent:.2f}')
    
    # Set the labels and title
    ax.set_xlabel('Year')
    ax.set_ylabel('Sea Level (inches)')
    ax.set_title('Rise in Sea Level')
    
    # Add legend
    ax.legend()
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    
    # Set reasonable axis limits
    ax.set_xlim(df['Year'].min() - 5, 2055)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot and return data (do not change)
    plt.savefig('sea_level_plot.png')
    return plt.gca()

# Additional function to show predictions
def show_predictions():
    """Display predictions for 2050 using both trend lines"""
    df = pd.read_csv('epa-sea-level.csv')
    
    # Calculate predictions for 2050
    # Using all data
    slope_all, intercept_all, _, _, _ = linregress(df['Year'], df['CSIRO Adjusted Sea Level'])
    prediction_all = slope_all * 2050 + intercept_all
    
    # Using data from 2000 onwards
    df_recent = df[df['Year'] >= 2000]
    slope_recent, intercept_recent, _, _, _ = linregress(df_recent['Year'], df_recent['CSIRO Adjusted Sea Level'])
    prediction_recent = slope_recent * 2050 + intercept_recent
    
    print("SEA LEVEL PREDICTIONS FOR 2050:")
    print("=" * 40)
    print(f"Using all data (1880-present): {prediction_all:.2f} inches")
    print(f"Using recent data (2000-present): {prediction_recent:.2f} inches")
    print(f"Difference: {abs(prediction_recent - prediction_all):.2f} inches")
    
    # Calculate rate of change
    print("\nRATE OF CHANGE:")
    print("=" * 40)
    print(f"All data: {slope_all:.4f} inches/year")
    print(f"Recent data: {slope_recent:.4f} inches/year")
    print(f"Acceleration factor: {slope_recent/slope_all:.2f}x")
    
    return prediction_all, prediction_recent

# Example usage and analysis
if __name__ == "__main__":
    # Load and examine the data
    df = pd.read_csv('epa-sea-level.csv')
    
    print("DATASET OVERVIEW:")
    print("=" * 40)
    print(f"Data shape: {df.shape}")
    print(f"Years covered: {df['Year'].min()} to {df['Year'].max()}")
    print(f"Total years: {df['Year'].max() - df['Year'].min() + 1}")
    
    print("\nFIRST FEW ROWS:")
    print(df.head())
    
    print("\nLAST FEW ROWS:")
    print(df.tail())
    
    print("\nBASIC STATISTICS:")
    print(df.describe())
    
    print("\nMISSING VALUES:")
    print(df.isnull().sum())
    
    # Create the plot
    print("\nCreating sea level plot...")
    ax = draw_plot()
    plt.show()
    
    # Show predictions
    print("\n")
    show_predictions()
    
    # Calculate some additional insights
    print("\nADDITIONAL INSIGHTS:")
    print("=" * 40)
    
    # Total rise from start to end
    total_rise = df['CSIRO Adjusted Sea Level'].iloc[-1] - df['CSIRO Adjusted Sea Level'].iloc[0]
    years_span = df['Year'].iloc[-1] - df['Year'].iloc[0]
    
    print(f"Total sea level rise from {df['Year'].iloc[0]} to {df['Year'].iloc[-1]}: {total_rise:.2f} inches")
    print(f"Average rate over {years_span} years: {total_rise/years_span:.4f} inches/year")
    
    # Recent trend (last 20 years)
    recent_20_years = df[df['Year'] >= df['Year'].max() - 19]
    if len(recent_20_years) >= 2:
        recent_rise = recent_20_years['CSIRO Adjusted Sea Level'].iloc[-1] - recent_20_years['CSIRO Adjusted Sea Level'].iloc[0]
        recent_years = recent_20_years['Year'].iloc[-1] - recent_20_years['Year'].iloc[0]
        print(f"Recent 20-year trend ({recent_20_years['Year'].iloc[0]}-{recent_20_years['Year'].iloc[-1]}): {recent_rise/recent_years:.4f} inches/year")
