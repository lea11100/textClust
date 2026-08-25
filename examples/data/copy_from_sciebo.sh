module load tools/rclone/1.66.0

dataset_stages=( "raw_data" "translated_no_emojis" "embedded" )

# Download the datasets from sciebo to the current directory
for s in "${dataset_stages[@]}"; do
    rclone copy -P sciebo:Data/textClust_stream_data/"$s" ./$s
done