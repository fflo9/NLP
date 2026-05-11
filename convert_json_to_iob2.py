import json
import os 

def convert_json_to_iob2(data, output_file):
    """
    Converts list of conversation dicts to a IOB2 formatted text file.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for item in data:
            text = item.get('text', '')
            denotations = item.get('denotations', [])
            
            # 1. Create a character-level map of the labels
            # We strip the metadata and keep just the base class (e.g., 'Symptom')
            char_label_map = {}
            for d in denotations:
                # Extract base label: 'BPerson,,flfo@itu.dk...' -> 'Person'
                raw_obj = d['obj'].split(',')[0]
                base_label = raw_obj[1:] if raw_obj.startswith(('B', 'I')) else raw_obj
                
                for i in range(d['span']['begin'], d['span']['end']):
                    char_label_map[i] = base_label

            # 2. Tokenization & Labeling
            # Using find() allows us to skip whitespace while staying aligned with character spans
            words = text.split()
            last_idx = 0
            prev_base_label = None

            for word in words:
                # Find the start index of this word in the original text
                start_idx = text.find(word, last_idx)
                end_idx = start_idx + len(word)
                last_idx = end_idx
                
                # Check the character map for a label at this position
                current_base_label = char_label_map.get(start_idx, "O")

                if current_base_label == "O":
                    tag = "O"
                    prev_base_label = None
                else:
                    # IOB2 Logic: B- for start of entity, I- if continuing same entity
                    if current_base_label == prev_base_label:
                        tag = f"I-{current_base_label}"
                    else:
                        tag = f"B-{current_base_label}"
                    prev_base_label = current_base_label
                
                # Write to file: word <space> tag
                f.write(f"{word} {tag}\n")
            
            # 3. Add a blank line after each conversation (sentence separator)
            f.write("\n")

if __name__ == "__main__":
    input_path = r"data\MTS_data.json"
    output_path = r"data_iob2\MTS_Data.iob2"

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        convert_json_to_iob2(data, output_path)
        
    except FileNotFoundError:
        print(f"Error: Could not find the file at {input_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")