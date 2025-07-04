import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Import data (Make sure to parse dates. Consider setting index column to 'date'.)
df = pd.read_csv('fcc-forum-pageviews.csv', parse_dates=['date'], index_col='date')

# Clean data by filtering out days when page views were in the top 2.5% or bottom 2.5%
df = df[
    (df['value'] >= df['value'].quantile(0.025)) &
    (df['value'] <= df['value'].quantile(0.975))
]

def draw_line_plot():
    # Draw line plot
    fig, ax = plt.subplots(figsize=(15, 5))
    
    # Create the line plot
    ax.plot(df.index, df['value'], color='red', linewidth=1)
    
    # Set title and labels
    ax.set_title('Daily freeCodeCamp Forum Page Views 5/2016-12/2019')
    ax.set_xlabel('Date')
    ax.set_ylabel('Page Views')
    
    # Improve the appearance
    plt.tight_layout()
    
    # Save image and return fig (don't change)
    fig.savefig('line_plot.png')
    return fig

def draw_bar_plot():
    # Copy and modify data for monthly bar plot
    df_bar = df.copy()
    
    # Extract year and month from the date index
    df_bar['year'] = df_bar.index.year
    df_bar['month'] = df_bar.index.month
    
    # Group by year and month, then calculate the mean
    df_bar = df_bar.groupby(['year', 'month'])['value'].mean().unstack()
    
    # Create the bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create bar plot
    df_bar.plot(kind='bar', ax=ax, legend=True)
    
    # Set title and labels
    ax.set_xlabel('Years')
    ax.set_ylabel('Average Page Views')
    ax.legend(title='Months', labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    # Save image and return fig (don't change)
    fig.savefig('bar_plot.png')
    return fig

def draw_box_plot():
    # Prepare data for box plots (this part is done!)
    df_box = df.copy()
    df_box.reset_index(inplace=True)
    df_box['year'] = [d.year for d in df_box.date]
    df_box['month'] = [d.strftime('%b') for d in df_box.date]
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Draw box plots (using Seaborn)
    # Year-wise box plot
    sns.boxplot(data=df_box, x='year', y='value', ax=axes[0])
    axes[0].set_title('Year-wise Box Plot (Trend)')
    axes[0].set_xlabel('Year')
    axes[0].set_ylabel('Page Views')
    
    # Month-wise box plot
    # Define the correct month order
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    sns.boxplot(data=df_box, x='month', y='value', ax=axes[1], order=month_order)
    axes[1].set_title('Month-wise Box Plot (Seasonality)')
    axes[1].set_xlabel('Month')
    axes[1].set_ylabel('Page Views')
    
    plt.tight_layout()
    
    # Save image and return fig (don't change)
    fig.savefig('box_plot.png')
    return fig

# Example usage and testing
if __name__ == "__main__":
    print("Data shape after cleaning:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nData info:")
    print(df.info())
    
    print("\nBasic statistics:")
    print(df.describe())
    
    print("\nDate range:")
    print(f"From: {df.index.min()} to {df.index.max()}")
    
    # Create all plots
    print("\nCreating line plot...")
    line_fig = draw_line_plot()
    plt.show()
    
    print("Creating bar plot...")
    bar_fig = draw_bar_plot()
    plt.show()
    
    print("Creating box plots...")
    box_fig = draw_box_plot()
    plt.show()
    
    print("\nAll plots have been created and saved!")
