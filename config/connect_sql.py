file_list = ["Monday-WorkingHours.csv", "Tuesday-WorkingHours.csv", "Wednesday-WorkingHours.csv", "Thursday-WorkingHours.csv"]

# Track if it's the very first batch of the very first file
is_first_upload = True

for file in file_list:
    print(f"Starting file: {file}")
    
    # Use chunksize inside read_csv to stream the file from disk
    # This returns an iterator instead of a full DataFrame
    reader = pd.read_csv(file, chunksize=50000, low_memory=False)
    
    for chunk in reader:
        # 1. Clean the chunk (same fixes as before)
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        chunk.columns = [c.strip().lower() for c in chunk.columns]
        
        # 2. Determine mode: 'replace' ONLY for the first chunk of the first file
        mode = 'replace' if is_first_upload else 'append'
        
        # 3. Upload the chunk
        chunk.to_sql(name='raw_sentiment_data', con=ENGINE, if_exists=mode, index=False)
        
        # After the first successful chunk, everything else MUST be 'append'
        is_first_upload = False
        
    print(f"Finished uploading all chunks of {file}")

print("\n--- ALL 1.4M ROWS SUCCESSFULLY STREAMED TO MYSQL ---")