import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# question (a) - delete "id" column
df = pd.read_csv('/Users/yijieren/Downloads/desktop/oregon/courses/AI_534_Fall_21/IA0/data_ia0.csv')
df.drop('id', axis=1, inplace=True)
# print(df)

# question (b) - split date into month, day, year
split_items = df["date"].str.split("/", n=2, expand=True)
df["month"] = split_items[0]
df["day"] = split_items[1]
df["year"] = split_items[2]
df.drop(columns=["date"], inplace=True)
# print(df)

# question (c)
#count unique values in columns of bedrooms, bathrooms, floors
unique_bedrooms_array = pd.unique(df["bedrooms"])
unique_bathrooms_array = pd.unique(df["bathrooms"])
unique_floors_array = pd.unique(df["floors"])
# print(unique_bedrooms_array)
# print(unique_bathrooms_array)
# print(unique_floors_array)

# boxplot for above 3 different features
plt.clf()
ax_bedrooms = sns.boxplot(x="bedrooms", y="price", data=df)
fig_bedrooms = ax_bedrooms.get_figure()
fig_bedrooms.savefig("./bedrooms_distinct_boxplot.jpg")
plt.clf()
ax_bathrooms = sns.boxplot(x="bathrooms", y="price", data=df)
fig_bathrooms = ax_bathrooms.get_figure()
fig_bathrooms.set_size_inches(10, 7.5)
fig_bathrooms.savefig("./bathrooms_distinct_boxplot.jpg")
plt.clf()
ax_floors = sns.boxplot(x="floors", y="price", data=df)
fig_floors = ax_floors.get_figure()
fig_floors.savefig("./floors_distinct_boxplot.jpg")

# question (d)
# generate covariance matrix
columns = ["sqft_living", "sqft_living15", "sqft_lot", "sqft_lot15"]
df_4_features = df[columns]
covariance_matrix = df_4_features.cov()
# print(covariance_matrix)

# scatter plot for sqrt_living vs sqrt_living15
plt.clf()
ax_sqrt_living = sns.scatterplot(data=df_4_features, x="sqft_living", y="sqft_living15")
fig_sqrt_living = ax_sqrt_living.get_figure()
fig_sqrt_living.savefig("./sqrt_living_scatterplot.jpg")

# scatter plot for sqrt_lot vs sqrt_lot15
plt.clf()
ax_sqrt_lot = sns.scatterplot(data=df_4_features, x="sqft_lot", y="sqft_lot15")
fig_sqrt_lot = ax_sqrt_lot.get_figure()
fig_sqrt_lot.savefig("./sqrt_lot_scatterplot.jpg")
