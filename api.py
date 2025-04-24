from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any
import uvicorn
import json

from models import EmailClassifier
from utils import PiiMasker, prepare_output

# Initialize FastAPI app
app = FastAPI(
    title="Email Classification API",
    description="API for classifying support emails with PII masking",
    version="1.0.0"
)

# Initialize components
pii_masker = PiiMasker()
email_classifier = EmailClassifier()

# Define request model
class EmailRequest(BaseModel):
    email: str

@app.post("/classify-email")
async def classify_email(request: EmailRequest) -> Dict[str, Any]:
    """
    Endpoint to classify an email while masking PII
    
    Args:
        request: Request containing the email text
        
    Returns:
        JSON response with classification and masked data
    """
    try:
        # Extract email text
        email_text = request.email
        
        # Mask PII
        masked_email, detected_entities = pii_masker.mask_pii(email_text)
        
        # Classify the masked email
        category = email_classifier.predict(masked_email)
        
        # Prepare the response
        response = prepare_output(
            input_email=email_text,
            masked_email=masked_email,
            detected_entities=detected_entities,
            category=category
        )
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Endpoint to check if the API is running"""
    return {"status": "healthy"}

def run_api(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the FastAPI application"""
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    run_api()