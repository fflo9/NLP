"""
BIO Tagging Implementation for NER (Named Entity Recognition)
==============================================================

This module implements BIO (Begin-Inside-Outside) tagging for entity recognition.
BIO tagging is a word-level annotation scheme that encodes entity boundaries through 
prefix annotations. Each token receives a label combining:
- Position indicator (B = Begin, I = Inside, O = Outside)
- Entity type (PERSON, LOCATION, MEDICAL_TERM)
"""

import pandas as pd
import json
import ast
from typing import List, Tuple, Dict
import re
from pathlib import Path


class BIOTagger:
    """
    A comprehensive BIO tagger for named entity recognition.
    
    This class handles the conversion of entity annotations (with character offsets)
    to BIO-tagged sequences at the word/token level.
    """
    
    # Define entity type mappings - map original labels to our target entities
    ENTITY_TYPE_MAPPING = {
        # Persons
        "T098": "PERSON",
        "T097": "PERSON",
        
        # Locations
        "T082": "LOCATION",
        
        # Organizations
        "T092": "ORGANIZATION",
        "T091": "ORGANIZATION",
        
        # Sympthom
        "T033": "SYMPTHOM",
        "T017": "SYMPTHOM",
        "T022": "SYMPTHOM",
        "T031": "SYMPTHOM",

        # History
        "T037": "HISTORY",
        "T201": "HISTORY",
        "T038": "HISTORY",
        "T005": "HISTORY",
        "T007": "HISTORY",
        "T103": "HISTORY",

        # Actions
        "T058": "ACTION",
        "T062": "ACTION",
        "T074": "ACTION",

        # Others
        "T204": "OTHER",
        "T170": "OTHER",
        "T168": "OTHER"

    }
    
    # Valid BIO labels that can be assigned
    VALID_LABELS = {
        "O",               # Outside - not part of any entity
        "B-PERSON",        # Begin person entity
        "I-PERSON",        # Inside person entity
        "B-LOCATION",
        "I-LOCATION", 
        "B-ORGANIZATION",  
        "I-ORGANIZATION",
        "B-SYMPTHOM",
        "I-SYMPTHOM",
        "B-HISTORY",
        "I-HISTORY",
        "B-ACTION",
        "I-ACTION",
        "B-OTHER",
        "I-OTHER"
    }
    
    def __init__(self):
        """Initialize the BIO tagger with default configuration."""
        self.verbose_output = True
    
    def tokenize_text(self, text: str) -> List[Tuple[int, int, str]]:
        """
        Tokenize text into words using whitespace and punctuation as delimiters.
        
        This function splits text into tokens and returns their character positions
        in the original text. This is crucial for aligning entity offsets with tokens.
        
        Args:
            text (str): The input text to tokenize
            
        Returns:
            List[Tuple[int, int, str]]: List of (start_position, end_position, token_text)
                - start_position: Character index where token begins in original text
                - end_position: Character index where token ends in original text
                - token_text: The actual token string
        
        Example:
            >>> tokenize_text("Where is Iguazu?")
            [(0, 5, 'Where'), (6, 8, 'is'), (9, 15, 'Iguazu'), (15, 16, '?')]
        """
        tokens_with_positions = []
        
        # Pattern to match tokens: words (alphanumeric, hyphens, dots) and punctuation
        # This regex captures contiguous word characters or individual punctuation
        token_pattern = r'\b[\w\-\.]+\b|[^\w\s]'
        
        for match in re.finditer(token_pattern, text):
            start_pos = match.start()
            end_pos = match.end()
            token_text = match.group()
            tokens_with_positions.append((start_pos, end_pos, token_text))
        
        return tokens_with_positions
    
    def get_entity_type(self, entity_type_code: str) -> str:
        """
        Map original entity type codes to our three target entity types.
        
        The MedMentions dataset uses UMLS type codes. This function maps those codes
        to our simplified entity types: PERSON, LOCATION, MEDICAL_TERM.
        
        Args:
            entity_type_code (str): The UMLS entity type code (e.g., "T098")
            
        Returns:
            str: One of {"PERSON", "LOCATION", "MEDICAL_TERM"} or None if not mapped
            
        Example:
            >>> get_entity_type("T098")
            "PERSON"
            >>> get_entity_type("T082")
            "LOCATION"
        """
        return self.ENTITY_TYPE_MAPPING.get(entity_type_code)
    
    def assign_bio_labels(self, tokens_with_positions: List[Tuple[int, int, str]], 
                         entities: List[Dict]) -> List[str]:
        """
        Assign BIO labels to each token based on entity annotations.
        
        The BIO tagging scheme works as follows:
        - B-TAG: First token of an entity (Begin)
        - I-TAG: Subsequent tokens of the same entity (Inside)
        - O: Token outside any entity (Outside)
        
        When an entity spans multiple tokens, the first token gets B-TAG and 
        subsequent tokens get I-TAG. This encodes entity boundaries.
        
        Args:
            tokens_with_positions (List[Tuple[int, int, str]]): Tokenized text with positions
            entities (List[Dict]): List of entities with their offsets and types
                Each entity dict should contain:
                - 'offsets': [[start, end]] - character position in original text
                - 'type': entity type code (e.g., "T098")
                
        Returns:
            List[str]: BIO label for each token in the same order as input tokens
        """
        # Initialize all labels as "O" (outside any entity)
        bio_labels = ["O"] * len(tokens_with_positions)
        
        # Process each entity in the document
        for entity in entities:
            # Extract entity information
            entity_offsets = entity.get("offsets", [[0, 0]])  # [[start, end]]
            entity_start, entity_end = entity_offsets[0]  # Get first span
            entity_type_code = entity.get("type", "")
            
            # Map the entity type code to our target types
            entity_type = self.get_entity_type(entity_type_code)
            
            # Skip if this entity type is not in our target set
            if entity_type is None:
                continue
            
            # Track whether we've labeled the first token of this entity
            is_first_token_of_entity = True
            
            # Check each token to see if it overlaps with the entity span
            for token_idx, (token_start, token_end, token_text) in enumerate(tokens_with_positions):
                # Check if token overlaps with entity span
                # Token overlaps if: token_start < entity_end AND token_end > entity_start
                if token_start < entity_end and token_end > entity_start:
                    if is_first_token_of_entity:
                        # First token of entity gets B- prefix
                        bio_labels[token_idx] = f"B-{entity_type}"
                        is_first_token_of_entity = False
                    else:
                        # Subsequent tokens of same entity get I- prefix
                        bio_labels[token_idx] = f"I-{entity_type}"
        
        return bio_labels
    
    def parse_entities_from_json(self, entities_json_str: str) -> List[Dict]:
        """
        Parse entity list from JSON string format.
        
        The CSV data contains entities as JSON strings. This function safely
        parses that JSON and extracts relevant entity information.
        
        Args:
            entities_json_str (str): JSON string representation of entities
                                     from the CSV
            
        Returns:
            List[Dict]: Parsed entity dictionaries with 'type', 'offsets', 'text'
            
        Raises:
            ValueError: If JSON is malformed or cannot be parsed
        """
        if not entities_json_str or pd.isna(entities_json_str):
            return []
        
        try:
            # Handle both string representation and actual strings
            if isinstance(entities_json_str, str):
                entities_list = ast.literal_eval(entities_json_str)
            else:
                entities_list = entities_json_str
            
            # Validate and extract relevant fields
            parsed_entities = []
            for entity in entities_list:
                if isinstance(entity, dict):
                    parsed_entities.append({
                        'type': entity.get('type'),
                        'offsets': entity.get('offsets', [[0, 0]]),
                        'text': entity.get('text', [''])[0] if entity.get('text') else ''
                    })
            
            return parsed_entities
        
        except (json.JSONDecodeError, ValueError, SyntaxError) as e:
            print(f"Warning: Could not parse entities: {str(e)}")
            return []
    
    def extract_passage_text(self, passages_str: str) -> str:
        """
        Extract and concatenate the actual text from passages.
        
        The 'passages' field contains document sections (title, abstract, etc.)
        with their corresponding text spans. This function extracts the combined text.
        
        Args:
            passages_str (str): JSON string containing passages with text
            
        Returns:
            str: Combined text from all passages
        """
        if not passages_str or pd.isna(passages_str):
            return ""
        
        try:
            passages_list = ast.literal_eval(passages_str)
            
            # Extract all text passages and join them
            full_text = ""
            current_offset = 0
            
            for passage in passages_list:
                if isinstance(passage, dict):
                    # Passages have 'text' as list of strings
                    passage_texts = passage.get('text', [])
                    if passage_texts:
                        # Get the full text of this passage
                        passage_text = ' '.join(passage_texts)
                        full_text += passage_text + " "
            
            return full_text.strip()
        
        except (json.JSONDecodeError, ValueError, SyntaxError) as e:
            print(f"Warning: Could not parse passages: {str(e)}")
            return ""
    
    def process_row(self, row: pd.Series) -> Dict:
        """
        Process a single row from the CSV and generate BIO-tagged output.
        
        This is the main processing function that orchestrates the entire
        BIO tagging pipeline for one document.
        
        Args:
            row (pd.Series): A row from the CSV DataFrame containing:
                - 'passages': Document passages with text
                - 'entities': Entity annotations with offsets and types
                
        Returns:
            Dict: Contains:
                - 'document_id': Unique document identifier
                - 'text': Full document text
                - 'tokens': List of token strings
                - 'bio_tags': Corresponding BIO label for each token
                - 'token_positions': Start/end positions for each token
        """
        # Step 1: Extract the full document text from passages
        document_text = self.extract_passage_text(row.get('passages', ''))
        
        # Step 2: Tokenize the document
        tokens_with_positions = self.tokenize_text(document_text)
        
        # Step 3: Parse entity annotations
        entities = self.parse_entities_from_json(row.get('entities', ''))
        
        # Step 4: Assign BIO labels to each token
        bio_labels = self.assign_bio_labels(tokens_with_positions, entities)
        
        # Extract tokens
        tokens = [token_text for _, _, token_text in tokens_with_positions]
        
        return {
            'document_id': row.get('id', 'unknown'),
            'text': document_text,
            'tokens': tokens,
            'bio_tags': bio_labels
        }
    

    def process_csv(self, csv_path: str) -> Tuple[List[Dict], pd.DataFrame]:
        """
        Process entire CSV file and generate BIO-tagged sequences.
        
        Args:
            csv_path (str): Path to the CSV file
            
        Returns:
            Tuple[List[Dict], pd.DataFrame]:
                - List of processed documents with BIO tags
                - DataFrame with the data
        """
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        processed_data = []
        
        # Process each row
        for idx, row in df.iterrows():
            try:
                processed_row = self.process_row(row)
                processed_data.append(processed_row)
                
                if self.verbose_output and idx % 100 == 0:
                    print(f"Processed {idx} documents...")
            
            except Exception as e:
                print(f"Error processing row {idx}: {str(e)}")
                continue
        
        return processed_data, df
    

    def save_bio_format(self, processed_data: List[Dict], output_path: str):
        """
        Save processed data in IOB2 format (standard format for NER datasets).
        
        The IOB2 format is a widely used standard for NER datasets:
        - Each line represents one token
        - Format: TOKEN\tBIO_TAG
        - Sentences are separated by blank lines
        - Document comments start with #
        
        Args:
            processed_data (List[Dict]): Processed documents with BIO tags
            output_path (str): Path to save the IOB2 format file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for doc_idx, doc in enumerate(processed_data):
                # Write document header
                f.write(f"# doc_id = {doc['document_id']}\n")
                f.write(f"# text = {doc['text'][:100]}...\n")
                
                # Write tokens with BIO tags
                for token, bio_tag in zip(doc['tokens'], doc['bio_tags']):
                    f.write(f"{token}\t{bio_tag}\n")
                
                # Blank line between documents
                f.write("\n")
        
        print(f"Saved IOB2 format to: {output_path}")
    

    def generate_statistics(self, processed_data: List[Dict]) -> Dict:
        """
        Generate statistics about the BIO-tagged dataset.
        
        Provides insight into entity distribution across the dataset,
        which is useful for understanding class balance.
        
        Args:
            processed_data (List[Dict]): Processed documents with BIO tags
            
        Returns:
            Dict: Statistics including entity counts and distributions
        """
        # Initialize counters
        entity_counts = {
            'O': 0,
            'B-PERSON': 0,
            'I-PERSON': 0,
            'B-LOCATION': 0,
            'I-LOCATION': 0,
            'B-ORGANIZATION': 0,
            'I-ORGANIZATION': 0,
            'B-SYMPTHOM': 0,
            'I-SYMPTHOM': 0,
            'B-HISTORY': 0,
            'I-HISTORY': 0,
            'B-ACTION': 0,
            'I-ACTION': 0,
            'B-OTHER': 0,
            'I-OTHER': 0
        }
        
        total_tokens = 0
        total_documents = 0
        
        # Count occurrences of each label
        for doc in processed_data:
            total_documents += 1
            for tag in doc['bio_tags']:
                total_tokens += 1
                if tag in entity_counts:
                    entity_counts[tag] += 1
        
        # Calculate statistics
        stats = {
            'total_documents': total_documents,
            'total_tokens': total_tokens,
            'token_counts': entity_counts,
            'percentages': {
                label: (count / total_tokens * 100) if total_tokens > 0 else 0
                for label, count in entity_counts.items()
            }
        }
        
        return stats
    
    def print_sample(self, processed_data: List[Dict], doc_index: int = 0):
        """
        Print a sample document with BIO tags for visualization and debugging.
        
        Args:
            processed_data (List[Dict]): Processed documents
            doc_index (int): Which document to print (default: first)
        """
        if doc_index >= len(processed_data):
            print("Document index out of range")
            return
        
        doc = processed_data[doc_index]
        
        print(f"\n{'='*80}")
        print(f"Document ID: {doc['document_id']}")
        print(f"Text: {doc['text'][:200]}...")
        print(f"\nToken-Level BIO Tags:")
        print(f"{'Token':<20} {'BIO Tag':<20}")
        print(f"{'-'*40}")
        
        for token, tag in zip(doc['tokens'][:30], doc['bio_tags'][:30]):  # Show first 30
            print(f"{token:<20} {tag:<20}")
        
        if len(doc['tokens']) > 30:
            print(f"... and {len(doc['tokens']) - 30} more tokens")


def main():
    """
    Main function to process all datasets and save as IOB2 format.
    """
    # Initialize the BIO tagger
    tagger = BIOTagger()
    
    print("="*80)
    print("BIO TAGGING - IOB2 FORMAT GENERATION")
    print("="*80)
 

    # Set up paths
    data_dir = Path("data")
    output_dir = Path("data_iob2")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}\n")
    
    # Process each dataset
    for split_name in ["train", "validation", "test"]:
        input_file = data_dir / f"{split_name}.csv"
        
        if not input_file.exists():
            print(f"Skipping {split_name} - file not found")
            continue
        
        print(f"Processing {split_name.upper()} set...")
        print(f"Input: {input_file}")
        
        # Process the CSV
        processed_data, df = tagger.process_csv(str(input_file))
        
        # Generate statistics
        stats = tagger.generate_statistics(processed_data)
        
        print(f"  Total documents: {stats['total_documents']}")
        print(f"  Total tokens: {stats['total_tokens']}")
        print(f"  Label Distribution:")
        for label, percentage in stats['percentages'].items():
            count = stats['token_counts'][label]
            print(f"    {label:<20}: {count:>6} tokens ({percentage:>6.2f}%)")
        
        # Save in IOB2 format
        output_file = output_dir / f"{split_name}.iob2"
        tagger.save_bio_format(processed_data, str(output_file))
        
        # Print sample from first document
        if processed_data:
            print(f"\nSample from first document:")
            tagger.print_sample(processed_data, 0)
        
        print()
    
    print("="*80)
    print("BIO TAGGING COMPLETE")
    print(f"All .iob2 files saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()