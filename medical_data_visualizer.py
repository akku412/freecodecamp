import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Import the data from medical_examination.csv and assign it to the df variable
df = pd.read_csv('medical_examination.csv')

# 2. Add an 'overweight' column to the data
# Calculate BMI and determine if person is overweight (BMI > 25)
df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2 > 25).astype(int)

# 3. Normalize data by making 0 always good and 1 always bad
# If cholesterol or gluc is 1, set to 0. If more than 1, set to 1.
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4. Draw the Categorical Plot
def draw_cat_plot():
    # 5. Create a DataFrame for the cat plot using pd.melt
    df_cat = pd.melt(df, 
                     id_vars=['cardio'], 
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    
    # 6. Group and reformat the data in df_cat to split it by cardio
    # Show the counts of each feature
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
    
    # 7. Convert the data into long format and create a chart using sns.catplot()
    fig = sns.catplot(data=df_cat, 
                      x='variable', 
                      y='total', 
                      hue='value', 
                      col='cardio', 
                      kind='bar')
    
    # 8. Get the figure for the output and store it in the fig variable
    fig = fig.fig
    
    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig

# 9. Draw the Heat Map
def draw_heat_map():
    # 10. Clean the data in the df_heat variable by filtering out incorrect data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &  # diastolic <= systolic
        (df['height'] >= df['height'].quantile(0.025)) &  # height >= 2.5th percentile
        (df['height'] <= df['height'].quantile(0.975)) &  # height <= 97.5th percentile
        (df['weight'] >= df['weight'].quantile(0.025)) &  # weight >= 2.5th percentile
        (df['weight'] <= df['weight'].quantile(0.975))    # weight <= 97.5th percentile
    ]
    
    # 11. Calculate the correlation matrix and store it in the corr variable
    corr = df_heat.corr()
    
    # 12. Generate a mask for the upper triangle and store it in the mask variable
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # 13. Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # 14. Plot the correlation matrix using sns.heatmap()
    sns.heatmap(corr, 
                mask=mask, 
                annot=True, 
                fmt='.1f', 
                center=0, 
                square=True, 
                cbar_kws={'shrink': 0.5})
    
    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig

# Example usage and testing
if __name__ == "__main__":
    # Test the functions
    print("Data shape:", df.shape)
    print("\nFirst few rows of processed data:")
    print(df.head())
    
    print("\nData types:")
    print(df.dtypes)
    
    print("\nOverweight distribution:")
    print(df['overweight'].value_counts())
    
    print("\nCholesterol distribution (normalized):")
    print(df['cholesterol'].value_counts())
    
    print("\nGlucose distribution (normalized):")
    print(df['gluc'].value_counts())
    
    # Create the plots
    print("\nCreating categorical plot...")
    cat_fig = draw_cat_plot()
    plt.show()
    
    print("Creating heat map...")
    heat_fig = draw_heat_map()
    plt.show()
