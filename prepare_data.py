import pandas as pd
import os

def create_final_dataset(output_filename="noise_data.csv"):
    station_month_file = os.path.join("data", "station_month.csv")
    stations_file = os.path.join("data", "stations.csv")

    if not os.path.exists("data") or not os.path.exists(station_month_file) or not os.path.exists(stations_file):
        print("--- ERROR: 'data' directory or source CSV files not found. ---")
        exit()

    print("Loading and merging data files...")
    station_month_df = pd.read_csv(station_month_file)
    stations_df = pd.read_csv(stations_file)

    station_month_df['Day'] = station_month_df['Day'].fillna(station_month_df['Day'].mean())
    station_month_df['Night'] = station_month_df['Night'].fillna(station_month_df['Night'].mean())

    merged_df = pd.merge(station_month_df, stations_df, on="Station", how="left")

    def categorize_noise(day_level, night_level):
        avg_noise = (day_level + night_level) / 2
        if avg_noise < 55: return "Low"
        elif avg_noise < 70: return "Medium"
        else: return "High"

    merged_df['NoiseLevel'] = merged_df.apply(lambda row: categorize_noise(row['Day'], row['Night']), axis=1)
    
    final_columns = ['Year', 'Month', 'Day', 'Night', 'City', 'State', 'Type', 'NoiseLevel']
    final_df = merged_df[final_columns].dropna()
    final_df.to_csv(output_filename, index=False)
    print(f"--- Successfully created '{output_filename}' ---")

if __name__ == '__main__':
    create_final_dataset()

