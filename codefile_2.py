import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("plots", exist_ok=True)

df = pd.read_csv("Daily Household Transactions.csv")
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
df.dropna(subset=['Date', 'Amount'], inplace=True)
df.drop_duplicates(inplace=True)

def save_and_show_plot(filename):
    plt.savefig(f"plots/{filename}", bbox_inches="tight")
    plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['Amount'], bins=50, kde=True)
plt.title("Distribution of Transaction Amounts")
plt.xlabel("Amount")
plt.ylabel("Frequency")
save_and_show_plot("distribution_amounts.png")

plt.figure(figsize=(12, 6))
sns.countplot(data=df, x="Category", order=df["Category"].value_counts().index)
plt.title("Transaction Counts by Category")
plt.xlabel("Category")
plt.ylabel("Count")
plt.xticks(rotation=45)
save_and_show_plot("transaction_counts_by_category.png")

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x="Income/Expense")
plt.title("Transaction Counts by Type")
plt.xlabel("Type")
plt.ylabel("Count")
save_and_show_plot("transaction_counts_by_type.png")

monthly_data = df.resample("M", on="Date").sum(numeric_only=True)
plt.figure(figsize=(14, 7))
plt.plot(monthly_data.index, monthly_data['Amount'], marker="o")
plt.title("Monthly Transaction Amounts")
plt.xlabel("Month")
plt.ylabel("Total Amount")
plt.grid(True)
save_and_show_plot("monthly_transaction_amounts.png")

daily_data = df.groupby(df["Date"].dt.date).sum(numeric_only=True)
plt.figure(figsize=(14, 7))
plt.plot(daily_data.index, daily_data['Amount'], marker="o")
plt.title("Daily Transaction Amounts")
plt.xlabel("Date")
plt.ylabel("Total Amount")
plt.grid(True)
save_and_show_plot("daily_transaction_amounts.png")

pivot_table = df.pivot_table(index="Date", columns="Category", values="Amount", aggfunc="sum", fill_value=0)
correlation_matrix = pivot_table.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap of Transaction Categories")
save_and_show_plot("correlation_heatmap.png")

plt.figure(figsize=(12, 8))
sns.boxplot(data=df, x="Amount", y="Category", order=df["Category"].value_counts().iloc[:5].index)
plt.title("Boxplot of Amount by Top 5 Categories")
save_and_show_plot("boxplot_top_categories.png")

plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="Income/Expense", y="Amount")
plt.title("Boxplot of Amount by Income/Expense")
save_and_show_plot("boxplot_income_expense.png")

total_income = df.loc[df["Income/Expense"] == "Income", "Amount"].sum()
total_expense = df.loc[df["Income/Expense"] == "Expense", "Amount"].sum()
net_savings = total_income - total_expense

with open("summary.txt", "w") as f:
    f.write(f"Total Income: {total_income:.2f}\n")
    f.write(f"Total Expense: {total_expense:.2f}\n")
    f.write(f"Net Savings: {net_savings:.2f}\n")

print("✅ Analysis complete. Plots saved in 'plots/' and summary in 'summary.txt'.")