import os
import argparse
import pandas as pd
from typing import Dict, Any

from models import EmailClassifier, train_model_from_data
from utils import PiiMasker, prepare_output
from api import run_api

def main() -> None:
    """Main function to run the email classification project"""
    parser = argparse.ArgumentParser(description="Email Classification System")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--data", type=str, default="c:/Users/Akhil/Desktop/PETTEM AMRUTH/Akaike/email_classification/Combined_emails_with_natural_pii.csv", help="Path to CSV")
    parser.add_argument("--api", action="store_true", help="Run the API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="API host")
    parser.add_argument("--port", type=int, default=8000, help="API port")
    parser.add_argument("--test", type=str, help="Test email for classification")
    
    args = parser.parse_args()
    
    # Train the model if requested
    if args.train:
        print(f"Training model with data from {args.data}")
        if not os.path.exists(args.data):
            print(f"Error: Data file {args.data} not found")
            return
        
        classifier = train_model_from_data(args.data)
        print("Model training complete")
    
    # Test classification if requested
    if args.test:
        print(f"Testing classification with email: {args.test}")
        classifier = EmailClassifier()
        try:
            classifier.load_model()
        except FileNotFoundError:
            print("Model not found. Please train the model first.")
            return
        
        pii_masker = PiiMasker()
        masked_email, detected_entities = pii_masker.mask_pii(args.test)
        category = classifier.predict(masked_email)
        
        result = prepare_output(
            input_email=args.test,
            masked_email=masked_email,
            detected_entities=detected_entities,
            category=category
        )
        
        print("Classification result:")
        print(f"Category: {result['category_of_the_email']}")
        print(f"Masked email: {result['masked_email']}")
        print(f"Detected entities: {len(result['list_of_masked_entities'])}")
    
    # Run the API if requested
    if args.api:
        print(f"Starting API server on {args.host}:{args.port}")
        run_api(host=args.host, port=args.port)

if __name__ == "__main__":
    main()