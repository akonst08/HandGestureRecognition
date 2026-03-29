import pandas as pd

# Load the CSV file
file_path = r'C:\Users\arest\Desktop\6th-SEMESTER\EPL445\Project\Group Project\model\keypoint_classifier\keypoint.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path, header=None)

# Keep only rows where the first column is an integer from 0 to 9
filtered_df = df[df[0].astype(str).str.match('^[0-4]$')]

# Save the filtered DataFrame to a new CSV file
filtered_file_path = r'C:\Users\arest\Desktop\6th-SEMESTER\EPL445\Project\Group Project\model\keypoint_classifier\keypoint.csv'
filtered_df.to_csv(filtered_file_path, index=False, header=False)

print(f"Filtered CSV saved to {filtered_file_path}")
