# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── default style ── 
sns.set_theme(style="whitegrid")
plt.rcParams["figure.facecolor"] = "#F8FAFC"

OUTPUT = "figures"

# Step 2: Load the Dataset
df = pd.read_csv("data/StudentPerformanceFactors.csv")

print("✅ Dataset loaded!")
print(f"   Rows    : {df.shape[0]}")
print(f"   Columns : {df.shape[1]}")
print()

# Step 3: First Look at the Data
print("── First 5 rows ──")
print(df.head())
print()

print("── Column names ──")
print(df.columns.tolist())
print()

print("── Data types ──")
print(df.dtypes)
print()

print("── Missing values ──")
print(df.isnull().sum())
print()

# Step 4: Basic Statistics using NumPy & Pandas
print("── Basic Statistics ──")
print(df.describe())
print()

# Using NumPy manually
scores = df["Exam_Score"].values  # convert to NumPy array

print("── NumPy Calculations on Exam Score ──")
print(f"   Mean   : {np.mean(scores):.2f}")
print(f"   Median : {np.median(scores):.2f}")
print(f"   Std Dev: {np.std(scores):.2f}")
print(f"   Min    : {np.min(scores)}")
print(f"   Max    : {np.max(scores)}")
print()

# Step 5: Data Cleaning
# Fill missing values with the most common value (mode)
df["Teacher_Quality"] = df["Teacher_Quality"].fillna(df["Teacher_Quality"].mode()[0])
df["Parental_Education_Level"] = df["Parental_Education_Level"].fillna(df["Parental_Education_Level"].mode()[0])
df["Distance_from_Home"] = df["Distance_from_Home"].fillna(df["Distance_from_Home"].mode()[0])

print("✅ Missing values filled!")
print(f"   Missing values left: {df.isnull().sum().sum()}")
print()

# Step 6: Add a New Column
# Create a Pass/Fail column (Pass if score >= 60)
df["Result"] = df["Exam_Score"].apply(lambda x: "Pass" if x >= 60 else "Fail")

print("── Pass / Fail Count ──")
print(df["Result"].value_counts())
print()

# FIGURE 1: Exam Score Overview (3 charts)
print("📊 Saving Figure 1: Score Overview...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Student Exam Score Overview", fontsize=16, fontweight="bold", y=1.02)

# Chart 1 — Histogram of Exam Scores
axes[0].hist(df["Exam_Score"], bins=20, color="#4C72B0", edgecolor="white")
axes[0].axvline(np.mean(scores), color="red", linestyle="--", label=f"Mean: {np.mean(scores):.1f}")
axes[0].set_title("Distribution of Exam Scores")
axes[0].set_xlabel("Exam Score")
axes[0].set_ylabel("Number of Students")
axes[0].legend()

# Chart 2 — Pass vs Fail Pie Chart
counts = df["Result"].value_counts()
axes[1].pie(counts, labels=counts.index, autopct="%1.1f%%",
            colors=["#68D391", "#FC8181"], startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[1].set_title("Pass vs Fail Rate")

# Chart 3 — Boxplot by Gender
sns.boxplot(data=df, x="Gender", y="Exam_Score", ax=axes[2], hue="Gender",
            palette={"Male": "#4C72B0", "Female": "#DD8452"}, legend=False)
axes[2].set_title("Exam Score by Gender")
axes[2].set_xlabel("")

plt.tight_layout()
plt.savefig(f"{OUTPUT}/fig1_score_overview.png", dpi=130, bbox_inches="tight")
plt.close()
print("   ✅ Saved fig1_score_overview.png")

# FIGURE 2: Study Habits (3 charts)
print("📊 Saving Figure 2: Study Habits...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("How Study Habits Affect Performance", fontsize=16, fontweight="bold", y=1.02)

# Chart 1 — Hours Studied vs Exam Score (scatter plot)
axes[0].scatter(df["Hours_Studied"], df["Exam_Score"],
                alpha=0.3, color="#4C72B0", s=15)
# Add trend line using NumPy
m, b = np.polyfit(df["Hours_Studied"], df["Exam_Score"], 1)
x_line = np.linspace(df["Hours_Studied"].min(), df["Hours_Studied"].max(), 100)
axes[0].plot(x_line, m * x_line + b, color="red", linewidth=2, label="Trend line")
axes[0].set_title("Hours Studied vs Exam Score")
axes[0].set_xlabel("Hours Studied")
axes[0].set_ylabel("Exam Score")
axes[0].legend()

# Chart 2 — Attendance vs Exam Score (scatter plot)
axes[1].scatter(df["Attendance"], df["Exam_Score"],
                alpha=0.3, color="#38A169", s=15)
m2, b2 = np.polyfit(df["Attendance"], df["Exam_Score"], 1)
x_line2 = np.linspace(df["Attendance"].min(), df["Attendance"].max(), 100)
axes[1].plot(x_line2, m2 * x_line2 + b2, color="red", linewidth=2, label="Trend line")
axes[1].set_title("Attendance % vs Exam Score")
axes[1].set_xlabel("Attendance (%)")
axes[1].set_ylabel("Exam Score")
axes[1].legend()

# Chart 3 — Sleep Hours vs Average Score (bar chart)
sleep_avg = df.groupby("Sleep_Hours")["Exam_Score"].mean()
axes[2].bar(sleep_avg.index, sleep_avg.values, color="#805AD5", edgecolor="white")
axes[2].set_title("Avg Score by Sleep Hours")
axes[2].set_xlabel("Sleep Hours per Night")
axes[2].set_ylabel("Average Exam Score")

plt.tight_layout()
plt.savefig(f"{OUTPUT}/fig2_study_habits.png", dpi=130, bbox_inches="tight")
plt.close()
print("   ✅ Saved fig2_study_habits.png")

# FIGURE 3: Family & Home Factors (3 charts)
print("📊 Saving Figure 3: Family Factors...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Family & Home Factors on Performance", fontsize=16, fontweight="bold", y=1.02)

# Chart 1 — Parental Involvement
order = ["Low", "Medium", "High"]
pi_avg = df.groupby("Parental_Involvement")["Exam_Score"].mean().reindex(order)
axes[0].bar(pi_avg.index, pi_avg.values,
            color=["#FC8181", "#F6AD55", "#68D391"], edgecolor="white")
for i, val in enumerate(pi_avg.values):
    axes[0].text(i, val + 0.1, f"{val:.1f}", ha="center", fontweight="bold")
axes[0].set_title("Avg Score by Parental Involvement")
axes[0].set_ylabel("Average Exam Score")
axes[0].set_ylim(60, 73)

# Chart 2 — Family Income
fi_avg = df.groupby("Family_Income")["Exam_Score"].mean().reindex(order)
axes[1].bar(fi_avg.index, fi_avg.values,
            color=["#FC8181", "#F6AD55", "#68D391"], edgecolor="white")
for i, val in enumerate(fi_avg.values):
    axes[1].text(i, val + 0.1, f"{val:.1f}", ha="center", fontweight="bold")
axes[1].set_title("Avg Score by Family Income")
axes[1].set_ylabel("Average Exam Score")
axes[1].set_ylim(60, 73)

# Chart 3 — Internet Access
internet_avg = df.groupby("Internet_Access")["Exam_Score"].mean()
axes[2].bar(internet_avg.index, internet_avg.values,
            color=["#68D391", "#FC8181"], edgecolor="white", width=0.4)
for i, val in enumerate(internet_avg.values):
    axes[2].text(i, val + 0.1, f"{val:.1f}", ha="center", fontweight="bold")
axes[2].set_title("Avg Score: Internet Access")
axes[2].set_ylabel("Average Exam Score")
axes[2].set_ylim(60, 73)

plt.tight_layout()
plt.savefig(f"{OUTPUT}/fig3_family_factors.png", dpi=130, bbox_inches="tight")
plt.close()
print("   ✅ Saved fig3_family_factors.png")

# FIGURE 4: Motivation & School (3 charts)
print("📊 Saving Figure 4: Motivation & School...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Motivation & School Environment", fontsize=16, fontweight="bold", y=1.02)

# Chart 1 — Motivation Level
mot_avg = df.groupby("Motivation_Level")["Exam_Score"].mean().reindex(order)
colors_m = ["#FC8181", "#F6AD55", "#68D391"]
axes[0].bar(mot_avg.index, mot_avg.values, color=colors_m, edgecolor="white")
for i, val in enumerate(mot_avg.values):
    axes[0].text(i, val + 0.05, f"{val:.1f}", ha="center", fontweight="bold")
axes[0].set_title("Avg Score by Motivation Level")
axes[0].set_ylabel("Average Exam Score")
axes[0].set_ylim(60, 73)

# Chart 2 — Teacher Quality
tq_avg = df.groupby("Teacher_Quality")["Exam_Score"].mean().reindex(order)
axes[1].bar(tq_avg.index, tq_avg.values, color=colors_m, edgecolor="white")
for i, val in enumerate(tq_avg.values):
    axes[1].text(i, val + 0.05, f"{val:.1f}", ha="center", fontweight="bold")
axes[1].set_title("Avg Score by Teacher Quality")
axes[1].set_ylabel("Average Exam Score")
axes[1].set_ylim(60, 73)

# Chart 3 — Peer Influence
peer_order = ["Negative", "Neutral", "Positive"]
peer_avg = df.groupby("Peer_Influence")["Exam_Score"].mean().reindex(peer_order)
axes[2].bar(peer_avg.index, peer_avg.values,
            color=["#FC8181", "#F6AD55", "#68D391"], edgecolor="white")
for i, val in enumerate(peer_avg.values):
    axes[2].text(i, val + 0.05, f"{val:.1f}", ha="center", fontweight="bold")
axes[2].set_title("Avg Score by Peer Influence")
axes[2].set_ylabel("Average Exam Score")
axes[2].set_ylim(60, 73)

plt.tight_layout()
plt.savefig(f"{OUTPUT}/fig4_motivation_school.png", dpi=130, bbox_inches="tight")
plt.close()
print("   ✅ Saved fig4_motivation_school.png")

# FIGURE 5: Correlation Heatmap
print("📊 Saving Figure 5: Correlation Heatmap...")

numeric_cols = ["Hours_Studied", "Attendance", "Sleep_Hours",
                "Previous_Scores", "Tutoring_Sessions", "Exam_Score"]

fig, ax = plt.subplots(figsize=(9, 7))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            ax=ax, linewidths=0.5, linecolor="white",
            center=0, vmin=-1, vmax=1)
ax.set_title("Correlation Between Numeric Features", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT}/fig5_correlation_heatmap.png", dpi=130, bbox_inches="tight")
plt.close()
print("   ✅ Saved fig5_correlation_heatmap.png")

# FIGURE 6: Attendance vs Exam Score — Deep Dive
print("📊 Saving Figure 6: Attendance Analysis...")

# Split attendance into 4 groups
df["Attendance_Group"] = pd.cut(
    df["Attendance"],
    bins=[59, 69, 79, 89, 100],
    labels=["Low\n(60–69%)", "Medium\n(70–79%)", "High\n(80–89%)", "Very High\n(90–100%)"]
)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Attendance vs Exam Score — Deep Dive Analysis",
             fontsize=17, fontweight="bold", y=1.01)
fig.patch.set_facecolor("#F8FAFC")
plt.subplots_adjust(hspace=0.45, wspace=0.35)

ATT_COLORS = ["#FC8181", "#F6AD55", "#68D391", "#4299E1"]

# Chart 1 — Scatter Plot with Trend Line
ax = axes[0, 0]
ax.scatter(df["Attendance"], df["Exam_Score"], alpha=0.25, color="#4C72B0", s=14)
m, b = np.polyfit(df["Attendance"], df["Exam_Score"], 1)
x_vals = np.linspace(df["Attendance"].min(), df["Attendance"].max(), 200)
ax.plot(x_vals, m * x_vals + b, color="red", linewidth=2.2,
        label=f"Trend line  r = {df['Attendance'].corr(df['Exam_Score']):.3f}")
ax.set_title("Attendance % vs Exam Score")
ax.set_xlabel("Attendance (%)")
ax.set_ylabel("Exam Score")
ax.legend(fontsize=9)

# Chart 2 — Average Score by Attendance Group
ax = axes[0, 1]
group_avg = df.groupby("Attendance_Group", observed=True)["Exam_Score"].mean()
bars = ax.bar(group_avg.index, group_avg.values, color=ATT_COLORS, edgecolor="white", width=0.6)
for bar, val in zip(bars, group_avg.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
            f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title("Avg Score by Attendance Group")
ax.set_ylabel("Average Exam Score")
ax.set_ylim(60, 75)

# Chart 3 — Boxplot by Attendance Group
ax = axes[0, 2]
groups_data = [
    df[df["Attendance_Group"] == g]["Exam_Score"].values
    for g in ["Low\n(60–69%)", "Medium\n(70–79%)", "High\n(80–89%)", "Very High\n(90–100%)"]
]
bp = ax.boxplot(groups_data, label=["Low", "Medium", "High", "Very High"],
                patch_artist=True, widths=0.5,
                medianprops={"color": "white", "linewidth": 2.5})
for patch, color in zip(bp["boxes"], ATT_COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)
ax.set_title("Score Spread by Attendance Group")
ax.set_xlabel("Attendance Group")
ax.set_ylabel("Exam Score")

# Chart 4 — Count of Students per Attendance Group (Pie)
ax = axes[1, 0]
group_counts = df["Attendance_Group"].value_counts().sort_index()
ax.pie(group_counts.values, labels=["Low", "Medium", "High", "Very High"],
       autopct="%1.1f%%", colors=ATT_COLORS, startangle=90,
       wedgeprops={"edgecolor": "white", "linewidth": 2})
ax.set_title("Students per Attendance Group")

# Chart 5 — Pass Rate by Attendance Group
ax = axes[1, 1]
pass_rate = df.groupby("Attendance_Group", observed=True).apply(
    lambda x: (x["Result"] == "Pass").mean() * 100, include_groups=False
)
bars = ax.bar(pass_rate.index, pass_rate.values, color=ATT_COLORS, edgecolor="white", width=0.6)
for bar, val in zip(bars, pass_rate.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title("Pass Rate by Attendance Group")
ax.set_ylabel("Pass Rate (%)")
ax.set_ylim(85, 102)

# Chart 6 — KDE Score Density per Attendance Group
ax = axes[1, 2]
label_map = {
    "Low\n(60–69%)":        ("Low (60–69%)",        "#FC8181"),
    "Medium\n(70–79%)":     ("Medium (70–79%)",      "#F6AD55"),
    "High\n(80–89%)":       ("High (80–89%)",        "#68D391"),
    "Very High\n(90–100%)": ("Very High (90–100%)",  "#4299E1"),
}
for group, (label, color) in label_map.items():
    subset = df[df["Attendance_Group"] == group]["Exam_Score"]
    if len(subset) > 1:
        subset.plot.kde(ax=ax, label=label, color=color, linewidth=2.2)
ax.set_title("Score Distribution by Attendance Group")
ax.set_xlabel("Exam Score")
ax.set_ylabel("Density")
ax.legend(fontsize=8.5)

plt.tight_layout()
plt.savefig(f"{OUTPUT}/fig6_attendance_analysis.png", dpi=130, bbox_inches="tight")
plt.close()
print("   ✅ Saved fig6_attendance_analysis.png")

# Print attendance stats
print()
print("── Attendance Analysis Summary ──")
print(f"   Correlation (Attendance vs Score) : {df['Attendance'].corr(df['Exam_Score']):.3f}")
print()
print("   Avg Score per Attendance Group:")
for group, val in group_avg.items():
    print(f"     {str(group).replace(chr(10),' '):<26}: {val:.2f}")
print()
print("   Pass Rate per Attendance Group:")
for group, val in pass_rate.items():
    print(f"     {str(group).replace(chr(10),' '):<26}: {val:.1f}%")

# Final Summary
print()
print("=" * 45)
print("  ANALYSIS COMPLETE — KEY FINDINGS")
print("=" * 45)
print(f"  Total Students : {len(df):,}")
print(f"  Average Score  : {np.mean(scores):.2f}")
print(f"  Pass Rate      : {(df['Result']=='Pass').mean()*100:.1f}%")
print()
print("  Correlation with Exam Score:")
corr_vals = df[numeric_cols].corr()["Exam_Score"].drop("Exam_Score").sort_values(ascending=False)
for feat, val in corr_vals.items():
    print(f"    {feat:<22}: {val:+.3f}")
print("=" * 45)
