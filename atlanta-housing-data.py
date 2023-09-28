import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# step 1: Create a mock dataset

data = {
    'SquareFeet': [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
    'Bedrooms': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Age': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Price': [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
}

df = pd.DataFrame(data)

# Step 2: Data Engineering - For simplicity, assume there are no missing values or outliers.

# Step 3: Linear Regression
features = ['SquareFeet', 'Bedrooms', 'Age']
target = 'Price'

x = df[features]
y = df[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)

# Adding predictions to the dataframe
df['PredictedPrice'] = model.predict(x)


# save to csv
df.to_csv('atlanta-housing-data.csv', index=False)