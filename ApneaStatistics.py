import xml.etree.ElementTree as ET
from pyedflib import EdfReader
import os
import pandas as pd
import numpy as np

def get_annotations(path:str, eventos = ['Obstructive Apnea', 'Central Apnea', 'Mixed Apnea']):
    tree = ET.parse(path)
    root = tree.getroot()

    annotations = {}

    scored_events = root.find('ScoredEvents')
    for evento in eventos:
        evento_annotations = []
        for scored_event in scored_events.findall('ScoredEvent'):
            name = scored_event.find('Name').text
            if name == evento:
                start = float(scored_event.find('Start').text)
                duration = float(scored_event.find('Duration').text)
                evento_annotations.append([start, duration])
        annotations[evento] = evento_annotations

    return annotations

path_annot = "C:/Users/elena/OneDrive/Documentos/TFG/Dataset/HomePAP/polysomnography/annotations-events-profusion/lab/full"
path_edf = "C:/Users/elena/OneDrive/Documentos/TFG/Dataset/HomePAP/polysomnography/edfs/lab/full"
csv_path = "C:/Users/elena/OneDrive/Documentos/TFG/Dataset/HomePAP/datasets/homepap-baseline-dataset-0.2.0.csv"

models_path = './models'
name0 = f'model_allfiles'
if not os.path.exists(models_path + '//' + name0):
    os.makedirs(models_path + '//' + name0)

files = [int(f.split('-')[3].split('.')[0]) for f in os.listdir(path_edf) if f.endswith('.edf')]

# Load the CSV file to map nsrrid with additional information
try:
    metadata_df = pd.read_csv(csv_path, delimiter=';', on_bad_lines='skip')  # Use 'on_bad_lines' to skip problematic rows
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit()

# Initialize a list to store signal statistics
signal_stats = []

for file in files:
    annotations = get_annotations(path_annot + f"\\homepap-lab-full-{str(file)}-profusion.xml")
    
    apnea_durations = [anot[1] for annotation in annotations for anot in annotations[annotation] if anot[1] > 0]
    apnea_count = len(apnea_durations)
    avg_apnea_duration = sum(apnea_durations) / len(apnea_durations) if apnea_durations else 0

    # Calculate apnea statistics for each type
    apnea_stats_by_type = {apnea_type: [anot[1] for anot in annotations[apnea_type]] for apnea_type in annotations}
    apnea_counts = {apnea_type: len(durations) for apnea_type, durations in apnea_stats_by_type.items()}
    avg_apnea_durations = {apnea_type: (sum(durations) / len(durations) if durations else 0) for apnea_type, durations in apnea_stats_by_type.items()}
    
    # Aggregate total apnea statistics
    total_apnea_count = sum(apnea_counts.values())
    total_avg_apnea_duration = sum([sum(durations) for durations in apnea_stats_by_type.values()]) / total_apnea_count if total_apnea_count > 0 else 0
    
    # Find the corresponding metadata for the file
    metadata_row = metadata_df[metadata_df['nsrrid'] == file]
    if not metadata_row.empty:
        age = metadata_row['age'].values[0]
        gender = metadata_row['gender'].values[0]
        ethnicity = metadata_row['ethnicity'].values[0]
        ahi = metadata_row['ahi'].values[0]
    else:
        age = gender = ethnicity = ahi = None  # Default to None if no match is found

    signal_stats.append({
        'File': f"homepap-lab-full-{str(file)}",
        'Total Apnea Count': total_apnea_count,
        'Total Average Apnea Duration (s)': total_avg_apnea_duration,
        'Obstructive Apnea Count': apnea_counts.get('Obstructive Apnea', 0),
        'Obstructive Average Duration (s)': avg_apnea_durations.get('Obstructive Apnea', 0),
        'Central Apnea Count': apnea_counts.get('Central Apnea', 0),
        'Central Average Duration (s)': avg_apnea_durations.get('Central Apnea', 0),
        'Mixed Apnea Count': apnea_counts.get('Mixed Apnea', 0),
        'Mixed Average Duration (s)': avg_apnea_durations.get('Mixed Apnea', 0),
        'Age': age,
        'Gender': gender,
        'Ethnicity': ethnicity,
        'AHI': ahi
    })

signal_stats_df = pd.DataFrame(signal_stats)
excel_output_path = 'signal_statistics.xlsx'
signal_stats_df.to_excel(excel_output_path, index=False)
print(f"Signal statistics saved to {excel_output_path}")