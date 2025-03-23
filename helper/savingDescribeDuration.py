import os
import pandas as pd

csv_file = os.path.join(".", "crema_d_dataset.csv")

df = pd.read_csv(csv_file)

if 'Duration' in df.columns:
    duration_stats = df['Duration'].describe()
    duration_stats_df = duration_stats.to_frame().reset_index()
    duration_stats_df.columns = ['Statistic', 'Value']  # Rename columns

    output_csv_file = os.path.join(".", "Analytics", "DescribeOfDurations.csv")
    duration_stats_df.to_csv(output_csv_file, index=False)

    print(f"Descriptive Statistics for Duration saved to {output_csv_file}")
else:
    print("Duration column not found in the DataFrame.")
