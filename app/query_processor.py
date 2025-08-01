import re
import spacy
from transformers import pipeline

class QueryProcessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.query_expander = pipeline("text2text-generation", 
                                     model="google/flan-t5-small")
        
        # Domain-specific abbreviation mappings
        self.abbreviations = {
            'M': 'male',
            'F': 'female',
            'yr': 'year',
            'yrs': 'years',
            'mo': 'month',
            'mos': 'months',
            'dx': 'diagnosis',
            'tx': 'treatment',
            'sx': 'surgery'
        }
        
        # Medical/Insurance terminology
        self.medical_terms = {
            'knee surgery': 'orthopedic knee procedure',
            'heart attack': 'myocardial infarction',
            'diabetes': 'diabetes mellitus'
        }
    
    def preprocess_query(self, query):
        """Clean and standardize the query"""
        # Convert to lowercase
        processed = query.lower()
        
        # Expand abbreviations
        for abbrev, full_form in self.abbreviations.items():
            processed = re.sub(r'\b' + abbrev.lower() + r'\b', full_form, processed)
        
        # Expand medical terms
        for term, expansion in self.medical_terms.items():
            processed = re.sub(r'\b' + term + r'\b', expansion, processed)
        
        return processed
    
    def extract_entities(self, query):
        """Extract named entities from query"""
        doc = self.nlp(query)
        entities = {
            'age': None,
            'location': None,
            'procedure': None,
            'time_period': None
        }
        
        # Extract age
        age_match = re.search(r'(\d+)[-\s]*(year|yr|y)[-\s]*old', query, re.IGNORECASE)
        if age_match:
            entities['age'] = age_match.group(1)
        
        # Extract entities using spaCy
        for ent in doc.ents:
            if ent.label_ == "GPE":  # Geopolitical entity (location)
                entities['location'] = ent.text
            elif ent.label_ == "DATE":
                entities['time_period'] = ent.text
        
        return entities
    
    def expand_query(self, query):
        """Generate expanded version of query for better retrieval"""
        entities = self.extract_entities(query)
        
        # Create structured query
        structured_parts = []
        if entities['age']:
            structured_parts.append(f"patient age {entities['age']} years")
        if entities['location']:
            structured_parts.append(f"treatment location {entities['location']}")
        
        expanded = f"{query}. " + " ".join(structured_parts)
        return expanded
