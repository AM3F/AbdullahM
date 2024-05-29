import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv('/content/birthsper1000 form 1960 to 2021.csv')

# Split the data into training and testing sets
X = df[['Year']]  # Features
y = df['BirthsPer(1000)']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)


# Plot the actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.title('Actual vs. Predicted BirthPer(1000)')
plt.xlabel('Year')
plt.ylabel('Births per 1000 people')
plt.legend()
plt.show()
