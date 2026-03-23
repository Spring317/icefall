import json
import os
import argparse
import soundfile as sf
from datasets import load_from_disk # Changed import

def main():
    # Added argparse so you can run it with a custom path if needed
    parser = argparse.ArgumentParser(description="Extract Bud500 Test Set to WAV")
    parser.add_argument("--dataset-path", type=str, default="/home/datasets/viet_bud500", 
                        help="Local path to the saved Hugging Face dataset")
    args = parser.parse_args()

    # 1. Load the dataset safely from local disk
    print(f"Loading the Bud500 dataset from local disk: {args.dataset_path}...")
    
    # This loads the DatasetDict instantly
    dataset = load_from_disk(args.dataset_path)
    
    # Grab just the test split
    test_data = dataset["test"]

    # 2. Setup Output Directory
    output_dir = "bud500_raw_test"
    audio_dir = os.path.join(output_dir, "wavs")
    os.makedirs(audio_dir, exist_ok=True)
    
    manifest_path = os.path.join(output_dir, "transcripts.jsonl")

    print(f"Extracting {len(test_data)} audio files to {audio_dir}...")
    
    # 3. Iterate, Save Audio, and Create Manifest
    with open(manifest_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(test_data):
            # Hugging Face Audio columns yield a dict with 'array' and 'sampling_rate'
            audio_data = item["audio"]
            
            # Extract text (handling potential variations in column names)
            text = item.get("transcription", item.get("sentence", ""))
            
            # Generate a clean ID for the file
            utt_id = f"bud500_test_{i:06d}"
            
            # Define save path
            wav_filename = f"{utt_id}.wav"
            wav_path = os.path.join(audio_dir, wav_filename)
            
            # Write the raw numpy array to a .wav file
            sf.write(
                file=wav_path, 
                data=audio_data["array"], 
                samplerate=audio_data["sampling_rate"]
            )
            
            # Save the metadata for your ASR evaluation
            meta = {
                "id": utt_id,
                "audio_filepath": os.path.abspath(wav_path),
                "text": text,
                "duration": len(audio_data["array"]) / audio_data["sampling_rate"]
            }
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

            if (i + 1) % 500 == 0:
                print(f"Processed {i + 1}/{len(test_data)} files...")

    print("-" * 50)
    print("Extraction Complete!")
    print(f"Raw audio saved to: {audio_dir}")
    print(f"Transcript manifest saved to: {manifest_path}")

if __name__ == "__main__":
    main()
