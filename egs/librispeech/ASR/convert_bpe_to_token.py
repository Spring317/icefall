import sys
import os

def convert_vocab(input_path, output_path):
    """
    Converts bpe.vocab to tokens.txt with strict rules:
    1. ID 0 becomes <blk> (Required for Transducer).
    2. [MASK] becomes <unk> (To satisfy export script requirement).
    3. Keeps total line count exactly 4000 (To match model weights).
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        return

    print(f"Reading '{input_path}'...")

    tokens = []
    
    with open(input_path, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) == 1:
                token = parts[0]
            else:
                token = " ".join(parts[:-1])
                
            tokens.append(token)

    print(f"Found {len(tokens)} tokens. converting...")

    with open(output_path, 'w', encoding='utf-8') as f_out:
        for i, token in enumerate(tokens):
            
            # 1. Force ID 0 to be <blk>
            if i == 0:
                final_token = "<blk>"
            
            # 2. Replace [MASK] with <unk> 
            # (We use [MASK] because it is usually unused in ASR inference)
            elif token == "[MASK]":
                final_token = "<unk>"
                
            # 3. Keep everything else the same
            else:
                final_token = token
            
            f_out.write(f"{final_token} {i}\n")

    print(f"Done! Created '{output_path}' with {len(tokens)} entries.")

if __name__ == "__main__":
    input_file = "bpe.vocab" 
    output_file = "tokens.txt"
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]

    convert_vocab(input_file, output_file)
