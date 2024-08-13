import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.stats import ttest_ind, shapiro, linregress

### HELPER FUNCTIONS ###
# Define a function to normalize the perceived_duration values within each group
def normalize_perceived_duration(group):
    min_val = group.min()
    max_val = group.max()
    return (group - min_val) / (max_val - min_val)



### DATA PREPROCESSING ###
# Read the CSV file into a DataFrame
df = pd.read_csv('research_data_rows.csv')

device_counts = df['device_identifier'].value_counts()
# Use this to filter device counts. Greater than 0 = include all. Useful for within-subject studies.
valid_devices = device_counts[device_counts > 0].index
df_filtered = df.copy()
df_filtered = df_filtered[df_filtered['device_identifier'].isin(valid_devices)]
df_filtered = df_filtered[df_filtered['valid'] == True]



### SUN SPEED vs. PERCEIVED DURATION ###
# Calculate correlation between SUN_SPEED and PERCEIVED_DURATION
correlation = df_filtered['sun_speed'].corr(df_filtered['perceived_duration'])
print(f"Correlation between sun_speed and perceived_duration: {correlation}")

# Group by sun_speed and calculate the average perceived_duration for each group, in case two entries happen to have the exact same time gain.
grouped_sun_speed = df_filtered.groupby('sun_speed')['perceived_duration'].mean()

# Plot the average perceived_duration against sun_speed
plt.figure(figsize=(10, 6))
plt.plot(grouped_sun_speed.index, grouped_sun_speed.values, marker='o', linestyle='-')
plt.xlabel('Time Gain')
plt.ylabel('Perceived Duration')
plt.title('Perceived Duration by Time Gain')

# Add trendline
coefficients = np.polyfit(grouped_sun_speed.index, grouped_sun_speed.values, 1)
trendline = np.poly1d(coefficients)
plt.plot(grouped_sun_speed.index, trendline(grouped_sun_speed.index), linestyle='--', color='red')

plt.grid(True)
plt.show()




### SUN SPEED vs. REACTION VALUE ###
# Calculate correlation between SUN_SPEED and REACTION_VALUE
correlation = df_filtered['sun_speed'].corr(df_filtered['reaction_value'])
print(f"Correlation between sun_speed and reaction_value: {correlation}")

# Group by sun_speed and calculate the average reaction_value for each group, in case two entries happen to have the exact same time gain.
grouped_sun_speed = df_filtered.groupby('sun_speed')['reaction_value'].mean()

# Normalize the values
normalized_grouped_reaction_value = (grouped_sun_speed-grouped_sun_speed.min())/(grouped_sun_speed.max()-grouped_sun_speed.min())

# Plot the average perceived_duration against sun_speed
plt.figure(figsize=(10, 6))
plt.plot(normalized_grouped_reaction_value.index, normalized_grouped_reaction_value.values, marker='o', linestyle='-')
plt.xlabel('Time Gain')
plt.ylabel('Normalized Reaction Value')
plt.title('Normalized Reaction Value by Time Gain')

# Add trendline
coefficients = np.polyfit(normalized_grouped_reaction_value.index, normalized_grouped_reaction_value.values, 1)
trendline = np.poly1d(coefficients)
plt.plot(normalized_grouped_reaction_value.index, trendline(grouped_sun_speed.index), linestyle='--', color='red')

plt.grid(True)
plt.show()


### SUN PERCEIVED_DURATION vs. REACTION VALUE ###
# Calculate correlation between PERCEIVED_DURATION and REACTION_VALUE
correlation = df_filtered['perceived_duration'].corr(df_filtered['reaction_value'])
print(f"Correlation between perceived_duration and reaction_value: {correlation}")

# Group by sun_speed and calculate the average reaction_value for each group, in case two entries happen to have the exact same time gain.
grouped_perceived_duration = df_filtered.groupby('perceived_duration')['reaction_value'].mean()

# Plot the average perceived_duration against sun_speed
plt.figure(figsize=(10, 6))
plt.plot(grouped_perceived_duration.index, grouped_perceived_duration.values, marker='o', linestyle='-')
plt.xlabel('Perceived Duration')
plt.ylabel('Reaction Value')
plt.title('Reaction Value by Perceived Duration')

# Add trendline
coefficients = np.polyfit(grouped_perceived_duration.index, grouped_perceived_duration.values, 1)
trendline = np.poly1d(coefficients)
plt.plot(grouped_perceived_duration.index, trendline(grouped_perceived_duration.index), linestyle='--', color='red')

plt.grid(True)
plt.show()



### CHECK FOR NORMAL DISTRIBUTION ###
# Check normality for perceived_duration
statistic, p_value_normality_pd = shapiro(df_filtered['perceived_duration'])

# Threshold for considering distribution normal
alpha = 0.05

print(p_value_normality_pd)
if p_value_normality_pd <= alpha:
    print("Data is not normally distributed. Student's t-test cannot be performed.")



### STATISTICAL SIGNIFICANCE OF SUN SPEED vs. PERCEIVED DURATION ###
# Split DataFrame based on sun_speed
df_low_sun_speed_duration = df_filtered[df_filtered['sun_speed'] < 1]['perceived_duration']
df_high_sun_speed_duration = df_filtered[df_filtered['sun_speed'] > 1]['perceived_duration']

# Perform Student's t-test
_, p_value = ttest_ind(df_low_sun_speed_duration, df_high_sun_speed_duration)

# Print p-value
print(f"P-value for SUN SPEED vs. PERCEIVED DURATION (Student's t-test): {p_value}")

# Check if correlation is statistically significant
alpha = 0.05
if p_value < alpha:
    print("The difference in perceived_duration between low and high sun_speed groups is statistically significant.")
else:
    print("There is no statistically significant difference in perceived_duration between low and high sun_speed groups.")




### STATISTICAL SIGNIFICANCE OF SUN SPEED vs. REACTION VALUE ###
# Split DataFrame based on sun_speed
df_low_sun_speed_reaction = df_filtered[df_filtered['sun_speed'] < 1]['reaction_value']
df_high_sun_speed_reaction = df_filtered[df_filtered['sun_speed'] > 1]['reaction_value']
# Perform Student's t-test
_, p_value = ttest_ind(df_low_sun_speed_reaction, df_high_sun_speed_reaction)

# Print p-value
print(f"P-value for SUN SPEED vs. REACTION (Student's t-test): {p_value}")

# Check if correlation is statistically significant
alpha = 0.05
if p_value < alpha:
    print("The difference in reaction_value between low and high sun_speed groups is statistically significant.")
else:
    print("There is no statistically significant difference in reaction_value between low and high sun_speed groups.")




### EVALUATION OF REACTION SPEED OVER TIME (TRENDS) ###
# Calculate trends
# Function to extract reactionTimes from JSON string
def extract_reaction_times(json_str):
    try:
        json_data = json.loads(json_str)
        return json_data.get('reactionTimes')
    except json.JSONDecodeError:
        return None

# Apply the function to create a new column
df_filtered['reaction_times'] = df_filtered['run_json'].apply(extract_reaction_times)

# Function to filter out inf and -inf values
def filter_inf(values):
    return [x for x in values if np.isfinite(x)]

# Function to filter out lapses as defined by Basner et al.
def filter_lapses(values):
    return [x for x in values if x < 500]

# Function to calculate linear trend coefficients and extract the first component
def extract_trend(values):
    # Fit a linear trend (polynomial of degree 1)
    coefficients = np.polyfit(range(len(values)), values, 1)
    # Extract the first component (coefficient of x)
    return coefficients[0]

# Apply the function to filter inf and -inf values and anomalies
df_filtered['reaction_times'] = df_filtered['reaction_times'].apply(filter_inf).apply(filter_lapses)

# Filter out rows with less than two elements in reaction_times to avoid edge cases in cases of errors in data.
df_filtered = df_filtered[df_filtered['reaction_times'].apply(len) > 1]

# Apply the function to extract trend coefficients
df_filtered['trend_coefficient'] = df_filtered['reaction_times'].apply(extract_trend)

# Group by sun_speed and calculate the average trend for each group
grouped_sun_speed = df_filtered.groupby('sun_speed')['trend_coefficient'].mean()

df_filtered = df_filtered.sort_values(by='sun_speed')
# Plot the average trend of reaction times against sun speed
plt.figure(figsize=(10, 6))
plt.plot(df_filtered['sun_speed'], df_filtered['trend_coefficient'], marker='o', linestyle='-')
plt.xlabel('Time Gain')
plt.ylabel('Reaction Trend')
plt.title('Reaction Trend by Time Gain')

# Add trendline
coefficients = np.polyfit(grouped_sun_speed.index, grouped_sun_speed.values, 1)
trendline = np.poly1d(coefficients)
plt.plot(grouped_sun_speed.index, trendline(grouped_sun_speed.index), linestyle='--', color='red')

plt.grid(True)
plt.show()

# Perform Student's t-test
result = linregress(df_filtered['sun_speed'], df_filtered['trend_coefficient'])
p_value = result.pvalue

# Print p-value
print(f"P-value for SUN SPEED vs. TREND (Linear Regression): {p_value}")

# Check if correlation is statistically significant
alpha = 0.05
if p_value < alpha:
    print("The correlation between sun speed and trend is statistically significant.")
else:
    print("There is no statistically significant correlation between sun speed and trend.")
