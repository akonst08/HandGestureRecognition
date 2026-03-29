import csv

# Define the frame range and gesture label
start_frame = 984
end_frame = 985
gesture_label = "Three 2"

# Write the CSV
with open("groundtruth1.csv", "a", newline="") as f:
    writer = csv.writer(f)
    #writer.writerow(["frame_id", "hand_sign"])  # CSV header
    for frame_id in range(start_frame, end_frame + 1):
        writer.writerow([frame_id, gesture_label])

print(f"CSV written for frames {start_frame} to {end_frame} with label '{gesture_label}'.")