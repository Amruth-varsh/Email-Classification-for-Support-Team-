import re
import json
from typing import Dict, List, Tuple, Any

class PiiMasker:
    """Class for masking and unmasking personally identifiable information (PII)"""
    
    def __init__(self):
        # Define regex patterns for different PII types
        self.patterns = {
            'full_name': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
            'email': r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone_number': r'\b(?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b',
            'dob': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            'aadhar_num': r'\b\d{4}[ -]?\d{4}[ -]?\d{4}\b',
            'credit_debit_no': r'\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b',
            'cvv_no': r'\bCVV:? \d{3,4}\b|\b\d{3,4} \(?CVV\)?\b',
            'expiry_no': r'\b(0[1-9]|1[0-2])[/-]\d{2}\b'
        }
        
        # Store original entities for later restoration
        self.entity_store = {}
        self.entity_counter = 0
    
    def _generate_entity_id(self) -> str:
        """Generate a unique ID for each entity to be masked"""
        entity_id = f"entity_{self.entity_counter}"
        self.entity_counter += 1
        return entity_id
    
    def mask_pii(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Mask PII in the given text
        
        Args:
            text: The input text to mask
            
        Returns:
            Tuple containing:
            - The masked text
            - List of detected entities with their positions and types
        """
        masked_text = text
        detected_entities = []
        
        # Reset entity store for this text
        self.entity_store = {}
        self.entity_counter = 0
        
        # Process each PII type
        for entity_type, pattern in self.patterns.items():
            matches = list(re.finditer(pattern, masked_text, re.IGNORECASE))
            
            # Process matches in reverse to avoid index shifting
            for match in reversed(matches):
                start, end = match.span()
                original_text = masked_text[start:end]
                
                # Store the original text for later restoration
                entity_id = self._generate_entity_id()
                self.entity_store[entity_id] = {
                    'original': original_text,
                    'type': entity_type,
                    'position': [start, end]
                }
                
                # Replace with mask
                masked_text = masked_text[:start] + f"[{entity_type}]" + masked_text[end:]
                
                # Add to detected entities list
                detected_entities.append({
                    "position": [start, end],
                    "classification": entity_type,
                    "entity": original_text
                })
        
        # Sort detected entities by position
        detected_entities.sort(key=lambda x: x["position"][0])
        
        return masked_text, detected_entities
    
    def unmask_pii(self, masked_text: str) -> str:
        """
        Restore the original PII in the masked text
        
        Args:
            masked_text: The masked text
            
        Returns:
            The original text with PII restored
        """
        # Currently not needed for the assignment as per the requirements
        # But implemented for completeness
        unmasked_text = masked_text
        
        for entity_id, entity_info in self.entity_store.items():
            mask = f"[{entity_info['type']}]"
            unmasked_text = unmasked_text.replace(mask, entity_info['original'], 1)
        
        return unmasked_text

def prepare_output(input_email: str, masked_email: str, 
                   detected_entities: List[Dict[str, Any]], 
                   category: str) -> Dict[str, Any]:
    """
    Prepare the standardized output format as required by the assignment
    
    Args:
        input_email: Original email text
        masked_email: Email with PII masked
        detected_entities: List of detected PII entities
        category: Classified category of the email
        
    Returns:
        JSON-formatted output
    """
    output = {
        "input_email_body": input_email,
        "list_of_masked_entities": detected_entities,
        "masked_email": masked_email,
        "category_of_the_email": category
    }
    
    return output