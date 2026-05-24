import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data.csv")

print(df.head())
print(df.describe())

# Pairplot
sns.pairplot(df, hue="gender")
plt.show()

# Heatmap
plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()

# Boxplots
plt.figure()
sns.boxplot(x="gender", y="math_score", data=df)
plt.show()

# Violin plot
plt.figure()
sns.violinplot(x="gender", y="reading_score", data=df)
plt.show()

# Regression plot
plt.figure()
sns.regplot(x="study_hours", y="math_score", data=df)
plt.show()

# Distribution plot
plt.figure()
sns.kdeplot(df["writing_score"], fill=True)
plt.show()

print("Advanced EDA Completed")
