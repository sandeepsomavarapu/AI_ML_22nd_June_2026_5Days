import pandas as pd



# Create a sample DataFrame
data = {
    "Name": ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 40],
}
print(data)
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)
print("Column Names:")
print(df.keys)
print("DataFrame Info:")
print(df.info())
print("DataFrame Description:")
print(df.describe())


