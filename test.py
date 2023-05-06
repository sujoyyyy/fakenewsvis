import os

# Define the file URL and output filename
file_url = "https://drive.google.com/uc?id=1IFQKH5TrhCGdqOXJ37ul5GIlqxkP9yv_&export=download"
filename = "bert_model.pt"

# Use wget to download the file
os.system(f"wget -O {filename} {file_url}")
