"""Project 5: Smart Factory Machine Maintenance Analysis (Tasks 1-15).

Put 05_machine_maintenance.csv in the same folder and run: python machine_maintenance_analysis.py
Reads the 1,500-record dataset and writes result tables, flagged records and
recommendations into the same folder (files start with "out_").
All findings are ASSOCIATIONS, not proof of causation.
"""
import numpy as np
import pandas as pd

INPUT = "Anant Bansal, 2026B03110135.csv" 
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)

SENSORS = ["Operating_Hours", "Temperature_C", "Vibration_mm_s",
           "Rotational_Speed_RPM", "Tool_Wear_Minutes"]


def head(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def save(df, name, index=True):
    df.to_csv("out_" + name, index=index)


# ---------------------------------------------------------------- Task 1
head("TASK 1: Load and inspect machine records")
# keep_default_na=False: the text "None" in Failure_Type is a real category, not missing
df = pd.read_csv(INPUT, keep_default_na=False, na_values=[""])
print("Shape:", df.shape)
print(df.head(), "\n")
df.info()
print("\nUnique values -> Maintenance_Required:", df["Maintenance_Required"].unique().tolist(),
      "| Failure_Type:", df["Failure_Type"].unique().tolist())

# ---------------------------------------------------------------- Task 2
head("TASK 2: Missing sensor readings and handling")
missing = df.isna().sum()
print(missing[missing > 0].to_frame("missing_count").assign(
    missing_pct=lambda x: (x["missing_count"] / len(df) * 100).round(2)))
# Handling: flag rows, then impute with the MEDIAN (robust to outliers). Only ~2% is missing,
# and dropping rows would discard otherwise-valid temperature/wear/hours readings.
df["Vibration_Imputed"] = df["Vibration_mm_s"].isna()
vib_median = df["Vibration_mm_s"].median()
df["Vibration_mm_s"] = df["Vibration_mm_s"].fillna(vib_median)
print(f"\nImputed {int(df['Vibration_Imputed'].sum())} vibration values with the median "
      f"({vib_median:.2f} mm/s). Remaining missing: {int(df[SENSORS].isna().sum().sum())}")

# ---------------------------------------------------------------- Task 3
head("TASK 3: Duplicate rows and machine identifiers")
print("Fully duplicated rows:", int(df.drop(columns="Vibration_Imputed").duplicated().sum()))
ids = df["Machine_ID"]
print("Missing Machine_ID:", int(ids.isna().sum()))
print("Distinct Machine_IDs:", ids.nunique())
print("IDs with odd format (not 'M' + 2 digits):", int((~ids.str.fullmatch(r"M\d{2}")).sum()))
print("Leading/trailing spaces:", int((ids != ids.str.strip()).sum()))
print("\nRecords per machine (repeat IDs are expected: many readings per machine):")
print(ids.value_counts().sort_index().to_string())

# ---------------------------------------------------------------- Task 4
head("TASK 4: Mean, median and standard deviation of sensor columns")
sensor_summary = df[SENSORS].agg(["mean", "median", "std"]).T.round(2)
print(sensor_summary)
save(sensor_summary, "task04_sensor_summary.csv")

# ---------------------------------------------------------------- Task 5
head("TASK 5: Maintenance-required counts and percentages")
mc = df["Maintenance_Required"].value_counts()
maint_summary = pd.DataFrame({"count": mc, "percent": (mc / len(df) * 100).round(2)})
print(maint_summary)
save(maint_summary, "task05_maintenance_counts.csv")

# ---------------------------------------------------------------- Task 6
head("TASK 6: Average temperature by maintenance status")
print(df.groupby("Maintenance_Required")["Temperature_C"].mean().round(2))
print("\nFull comparison (groupby(...).mean(numeric_only=True)):")
status_means = df.groupby("Maintenance_Required").mean(numeric_only=True).round(2)
print(status_means)
save(status_means, "task06_means_by_maintenance_status.csv")

# ---------------------------------------------------------------- Task 7
head("TASK 7: Vibration, maintenance-required vs other records")
vib_cmp = df.groupby("Maintenance_Required")["Vibration_mm_s"].agg(
    ["count", "mean", "median", "std", "max"]).round(2)
print(vib_cmp)
diff = vib_cmp.loc["Yes", "mean"] - vib_cmp.loc["No", "mean"]
print(f"\nMean vibration difference (Yes - No): {diff:+.2f} mm/s")
save(vib_cmp, "task07_vibration_by_status.csv")

# ---------------------------------------------------------------- Task 8
head("TASK 8: Machines ranked by average operating hours")
ranking = (df.groupby("Machine_ID")["Operating_Hours"].agg(["mean", "count"])
           .round(1).sort_values("mean", ascending=False))
ranking.columns = ["avg_operating_hours", "records"]
ranking.insert(0, "rank", range(1, len(ranking) + 1))
print(ranking)
save(ranking, "task08_machine_ranking.csv")

# ---------------------------------------------------------------- Task 9
head("TASK 9: Unusually high vibration using IQR")
q1, q3 = df["Vibration_mm_s"].quantile([0.25, 0.75])
iqr = q3 - q1
upper = q3 + 1.5 * iqr
lower = q1 - 1.5 * iqr
print(f"Q1={q1:.2f}  Q3={q3:.2f}  IQR={iqr:.2f}  upper fence={upper:.2f}  lower fence={lower:.2f}")
high_vib = df[df["Vibration_mm_s"] > upper].sort_values("Vibration_mm_s", ascending=False)
print(f"Records above upper fence: {len(high_vib)} ({len(high_vib)/len(df)*100:.2f}%)")
print(high_vib.head(10).to_string())
save(high_vib, "task09_high_vibration_records.csv", index=False)

# ---------------------------------------------------------------- Task 10
head("TASK 10: Group records by Failure_Type")
ft = df.groupby("Failure_Type").agg(
    records=("Machine_ID", "size"),
    maintenance_yes=("Maintenance_Required", lambda s: (s == "Yes").sum()),
).assign(
    pct_of_records=lambda x: (x["records"] / len(df) * 100).round(2),
    maintenance_rate_pct=lambda x: (x["maintenance_yes"] / x["records"] * 100).round(2),
)
print(ft)
save(ft, "task10_failure_type_groups.csv")

# ---------------------------------------------------------------- Task 11
head("TASK 11: Average tool wear by failure type")
wear_by_ft = (df.groupby("Failure_Type")["Tool_Wear_Minutes"]
              .agg(["mean", "median", "count"]).round(2).sort_values("mean", ascending=False))
print(wear_by_ft)
save(wear_by_ft, "task11_tool_wear_by_failure_type.csv")

# ---------------------------------------------------------------- Task 12
head("TASK 12: NumPy Boolean indexing - high-temperature records")
TEMP_THRESHOLD = 85.0  # documented choice: roughly mean + 1.2 SD, a common caution level
temps = df["Temperature_C"].to_numpy()
mask = temps > TEMP_THRESHOLD        # Boolean mask
hot_temps = temps[mask]              # Boolean indexing on the NumPy array
print(f"Threshold: > {TEMP_THRESHOLD} C")
print(f"High-temperature records: {mask.sum()} ({mask.mean()*100:.2f}%)")
print(f"Mean of flagged temperatures: {hot_temps.mean():.2f} C | max: {hot_temps.max():.1f} C")
hot_df = df.loc[mask]
print("Maintenance rate among hot records:",
      f"{(hot_df['Maintenance_Required'] == 'Yes').mean()*100:.2f}%",
      "| overall:", f"{(df['Maintenance_Required'] == 'Yes').mean()*100:.2f}%")
save(hot_df, "task12_high_temperature_records.csv", index=False)

# ---------------------------------------------------------------- Task 13
head("TASK 13: Risk band with documented thresholds")
THRESH = {
    "Temperature_C": 85.0,               # high operating temperature
    "Vibration_mm_s": round(upper, 2),   # IQR upper fence from Task 9
    "Tool_Wear_Minutes": 250,            # tool near end of its 0-299 min range
    "Operating_Hours": 9000,             # very high accumulated usage
}
print("Thresholds (reading > threshold counts as 1 flag):", THRESH)
score = sum((df[c].to_numpy() > t).astype(int) for c, t in THRESH.items())
df["Risk_Score"] = score
df["Risk_Band"] = np.where(score >= 2, "High", np.where(score == 1, "Medium", "Low"))
print("Rule: High = 2+ flags; Medium = 1 flag; Low = 0 flags\n")
band = df.groupby("Risk_Band").agg(
    records=("Machine_ID", "size"),
    maintenance_yes=("Maintenance_Required", lambda s: (s == "Yes").sum()),
).reindex(["High", "Medium", "Low"])
band["pct_of_records"] = (band["records"] / len(df) * 100).round(2)
band["maintenance_rate_pct"] = (band["maintenance_yes"] / band["records"] * 100).round(2)
print(band)
save(band, "task13_risk_bands.csv")

# ---------------------------------------------------------------- Task 14
head("TASK 14: Correlations among numeric sensor columns")
corr = df[SENSORS].corr().round(3)
print(corr)
pairs = (corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
         .rename("r").reset_index())
pairs["abs_r"] = pairs["r"].abs()
print("\nStrongest pair:", pairs.sort_values("abs_r", ascending=False).iloc[0].to_dict())
save(corr, "task14_correlation_matrix.csv")

# ---------------------------------------------------------------- Deliverables
flagged = df[(df["Risk_Band"] == "High") | (df["Vibration_mm_s"] > upper)
             | (df["Temperature_C"] > TEMP_THRESHOLD)]
flagged = flagged.sort_values(["Risk_Score", "Vibration_mm_s"], ascending=False)
save(flagged, "flagged_records.csv", index=False)
save(df, "cleaned_with_risk_bands.csv", index=False)

# ---------------------------------------------------------------- Task 15
head("TASK 15: Five recommendations for maintenance teams")
yes_rate = (df["Maintenance_Required"] == "Yes").mean() * 100
hi_rate = band.loc["High", "maintenance_rate_pct"]
lo_rate = band.loc["Low", "maintenance_rate_pct"]
top_ops, top_ops_hrs = ranking.index[0], ranking.iloc[0]["avg_operating_hours"]
recs = [
    f"1. Triage with the risk band. {len(flagged)} records are in flagged_records.csv. Review "
    f"High-band readings first, and re-tune thresholds as more data arrives. (In this data the "
    f"maintenance rate is {hi_rate}% in the High band vs {lo_rate}% in the Low band.)",
    f"2. Inspect high-vibration readings. {len(high_vib)} records exceed the IQR fence of "
    f"{upper:.2f} mm/s. Schedule vibration checks (bearings, balance, mounting) for them, "
    f"even though mean vibration differs by only {diff:+.2f} mm/s between maintenance groups.",
    f"3. Watch temperature. {int(mask.sum())} readings exceed {TEMP_THRESHOLD:.0f} C. Add an alert "
    f"at this level and check cooling and lubrication, especially for Overheat-type failures.",
    f"4. Prioritise heavily used machines. {top_ops} has the highest average operating hours "
    f"({top_ops_hrs:.0f} h). Use hours-based inspection schedules for the top-ranked machines.",
    f"5. Improve data quality and collect more context. Vibration had missing readings "
    f"(median-imputed and flagged), so add sensor-health checks. Sensor correlations are weak "
    f"(max |r| = {pairs['abs_r'].max():.2f}) and only {yes_rate:.1f}% of records need maintenance, so "
    f"record load, machine age and last-service date before relying on any single sensor.",
]
for r in recs:
    print(r + "\n")
with open("out_task15_recommendations.txt", "w") as f:
    f.write("\n\n".join(recs))
print("NOTE: all patterns above are associations, not proof of causation.")
print("\nOutput files written to the current folder (names start with out_).")
