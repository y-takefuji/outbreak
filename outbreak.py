import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# Read the CSV file
df = pd.read_csv('NORS_20250328.csv')

# Convert Year and Month to datetime
df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(Day=1))

# Filter for Norovirus cases
df['is_norovirus'] = df['Etiology'].str.contains('Norovirus', na=False)
norovirus_data = df[df['is_norovirus']]

# First Graph: Overall monthly trends
monthly_stats = norovirus_data.groupby('Date').agg({
    'Illnesses': 'sum',
    'Info On Deaths': 'sum'
}).reset_index()

plt.figure(figsize=(15, 6), dpi=300)
plt.plot(monthly_stats['Date'], monthly_stats['Illnesses'], 'k:', label='Illnesses')
plt.plot(monthly_stats['Date'], monthly_stats['Info On Deaths'], 'k-', label='Deaths')
plt.xticks(rotation=90)
plt.xlabel('Year-Month')
plt.ylabel('Count')
plt.title('Monthly Norovirus Illnesses and Deaths')
plt.legend()
plt.tight_layout()
plt.savefig('monthly_overall_trends.jpg', bbox_inches='tight', dpi=300)
plt.show()
plt.close()

# Second Graph: Monthly deaths by transmission mode
modes = ['Food', 'Person-to-person', 'Water']
monthly_mode_stats = norovirus_data[norovirus_data['Primary Mode'].isin(modes)].groupby(['Date', 'Primary Mode'])['Info On Deaths'].sum().reset_index()

plt.figure(figsize=(15, 6), dpi=300)
for mode, style in zip(modes, ['-', '--', ':']):
    mode_data = monthly_mode_stats[monthly_mode_stats['Primary Mode'] == mode]
    plt.plot(mode_data['Date'], mode_data['Info On Deaths'], style, color='black', label=mode)

plt.xticks(rotation=90)
plt.xlabel('Year-Month')
plt.ylabel('Number of Deaths')
plt.title('Monthly Norovirus Deaths by Transmission Mode')
plt.legend()
plt.tight_layout()
plt.savefig('monthly_deaths_by_mode.jpg', bbox_inches='tight', dpi=300)
plt.show()
plt.close()

# Third Graph: Person-to-person annual deaths with predictions
person_to_person = norovirus_data[norovirus_data['Primary Mode'] == 'Person-to-person']
annual_deaths = person_to_person.groupby('Year')['Info On Deaths'].sum().reset_index()

# Filter data for display and regression
display_data = annual_deaths[annual_deaths['Year'].between(2008, 2023)]
recent_data = annual_deaths[annual_deaths['Year'].between(2021, 2023)]

# Prepare regression data
X = recent_data['Year'].values.reshape(-1, 1)
y = recent_data['Info On Deaths'].values

# Fit model
model = LinearRegression()
model.fit(X, y)
r_squared = model.score(X, y)

# Make predictions
prediction_years = np.array([[2023], [2024], [2025]])
predictions = model.predict(prediction_years)

# Create third plot
plt.figure(figsize=(10, 6), dpi=300)
plt.plot(display_data['Year'], display_data['Info On Deaths'], 'k-', label='Historical Deaths')
plt.plot(prediction_years.flatten(), predictions, 'k--', label='Predicted Deaths')
plt.xlabel('Year')
plt.ylabel('Number of Deaths')
plt.title('Annual Person-to-Person Norovirus Deaths with Predictions (2021-2023 based)')
plt.legend()
plt.grid(True)

# Add regression details to plot
equation = f'y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}'
r2_text = f'R² = {r_squared:.3f}'
plt.text(0.02, 0.98, equation + '\n' + r2_text, 
         transform=plt.gca().transAxes, 
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.savefig('annual_deaths_prediction.jpg', bbox_inches='tight', dpi=300)
plt.show()
plt.close()

# Print prediction results
print("\nModel Details:")
print(f"Slope: {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"R-squared: {r_squared:.3f}")

print("\nPredicted deaths for:")
print(f"2023: {predictions[0]:.0f} (Actual: {recent_data.iloc[-1]['Info On Deaths']:.0f})")
print(f"2024: {predictions[1]:.0f}")
print(f"2025: {predictions[2]:.0f}")

# Print actual values for regression period
print("\nActual deaths for regression period (2021-2023):")
for _, row in recent_data.iterrows():
    print(f"{int(row['Year'])}: {row['Info On Deaths']:.0f}")
