"""
HR and analytics Project

"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


file_path = "HRDataset_v14.csv"  # Change path if needed
df = pd.read_csv(file_path)

# Data Cleaning
# Parse date columns
date_cols = ["DOB", "DateofHire", "DateofTermination", "LastPerformanceReview_Date"]
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors="coerce")

# Standardize text columns
text_cols = ["Department", "Position", "MaritalDesc", "CitizenDesc", "RaceDesc",
             "Sex", "PerformanceScore", "EmploymentStatus"]
for col in text_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# Fix numeric columns
num_cols = ["Salary", "Absences", "EngagementSurvey", "EmpSatisfaction"]
for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Feature Engineering
today = pd.Timestamp("2019-12-31")

# Age and Tenure
df["Age"] = ((today - df["DOB"]).dt.days / 365.25).round(1)
df["TenureYears"] = ((df["DateofTermination"].fillna(today) - df["DateofHire"]).dt.days / 365.25).round(2)

# Termination flag
df["IsTerminated"] = df["Termd"].astype(int)

# Salary per special project
df["ProjectsPlus1"] = df["SpecialProjectsCount"].astype(float) + 1.0
df["Salary_per_Project"] = df["Salary"] / df["ProjectsPlus1"]

# Numeric performance score
mapping = {"Exceeds": 4, "Fully Meets": 3, "Needs Improvement": 2, "PIP": 1}
df["PerfScoreNum"] = df["PerformanceScore"].map(mapping)

# Z-score of salary within position (NumPy calculation)
df["Salary_z_within_position"] = df.groupby("Position")["Salary"].transform(
    lambda s: (s - s.mean()) / s.std(ddof=0)
)

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(8, 5))
df["Salary"].hist(bins=30, color="orange")
plt.title("Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Count")
plt.show()

# Average salary by department
dept_salary = df.groupby("Department")["Salary"].mean().sort_values(ascending=False)
dept_salary.plot(kind="bar", color="orange", title="Average Salary by Department")
plt.ylabel("Average Salary")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Attrition rate by department
attr_rate = df.groupby("Department")["IsTerminated"].mean().sort_values(ascending=False)
attr_rate.plot(kind="bar", color="orange", title="Attrition Rate by Department")
plt.ylabel("Attrition Rate")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Engagement vs Performance
plt.scatter(df["EngagementSurvey"], df["PerfScoreNum"], color="orange")
plt.title("Engagement vs Performance")
plt.xlabel("Engagement Survey")
plt.ylabel("Performance Score (numeric)")
plt.show()

# Correlation heatmap for selected features
features = ["Salary","Age","TenureYears","EmpSatisfaction",
            "EngagementSurvey","SpecialProjectsCount","Absences","PerfScoreNum"]
corr = df[features].corr()
plt.imshow(corr, cmap="viridis", interpolation="nearest")
plt.title("Correlation Heatmap (Selected Features)")
plt.xticks(range(len(features)), features, rotation=45, ha="right")
plt.yticks(range(len(features)), features)
plt.colorbar()
plt.show()

# Quantitative Insights
insights = {
    "Total Employees": len(df),
    "Active Employees": int((df["IsTerminated"] == 0).sum()),
    "Attrition Rate": df["IsTerminated"].mean(),
    "Median Salary": np.median(df["Salary"]),
    "Top 3 Departments by Avg Salary": dept_salary.head(3).to_dict(),
    "Top 3 Departments by Attrition": attr_rate.head(3).to_dict(),
    "Engagement vs Performance Corr": np.corrcoef(df["EngagementSurvey"], df["PerfScoreNum"])[0, 1],
    "Tenure vs Attrition Corr": np.corrcoef(df["TenureYears"], df["IsTerminated"])[0, 1],
    "Salary Outlier % (|z|>2)": (df["Salary_z_within_position"].abs() > 2).mean(),
    "Absence vs Performance Corr": np.corrcoef(df["Absences"], df["PerfScoreNum"])[0, 1],
}

print("\n=== KEY INSIGHTS ===")
for k, v in insights.items():
    print(f"{k:35s}: {v}")

# Department Summary
dept_summary = df.groupby("Department").agg(
    Headcount=("EmpID", "count"),
    Avg_Salary=("Salary", "mean"),
    Attrition_Rate=("IsTerminated", "mean"),
    Avg_Performance=("PerfScoreNum", "mean"),
    Avg_Tenure_Years=("TenureYears", "mean")
).round(2)

print("\n=== Department Summary ===")
print(dept_summary)

#Save Cleaned Dataset
df.to_csv("HRDataset_v14_engineered.csv", index=False)
print("\nSaved cleaned & feature-engineered dataset to HRDataset_v14_engineered.csv")
