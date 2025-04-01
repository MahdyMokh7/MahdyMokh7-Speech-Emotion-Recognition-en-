import pandas as pd
import os

def count_actors(input_file, output_file):
    df = pd.read_csv(input_file)
    
    male_count = df[df['Sex'] == 'Male']['Actor ID'].nunique()
    female_count = df[df['Sex'] == 'Female']['Actor ID'].nunique()
    
    count_df = pd.DataFrame({'Male': [male_count], 'Female': [female_count]})
    
    count_df.to_csv(output_file, index=False)
    
    print(f"Counts saved to {output_file}")

count_actors(os.path.join("Analytics", "crema_d_dataset.csv"), os.path.join("Analytics", "actor_counts.csv"))
