# Email Classification System

This project implements an email classification system for a company's support team. The system categorizes incoming support emails into predefined categories while masking personal information (PII) before processing.

## Features

- Email classification into support categories
- PII masking and detection without using LLMs
- RESTful API for processing emails
- Configurable and extensible architecture

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/email-classification.git
   cd email-classification
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

Before using the system, you need to train the classification model with your dataset:

```
python app.py --train --data path/to/your/data.csv
```

Your data should be a CSV file with at least two columns: `email` and `type`, where `email` contains the email text and `type` contains the category.

### Running the API

Start the API server:

```
python app.py --api --host 0.0.0.0 --port 8000
```

### Testing the Classification

You can test the classification directly from the command line:

```
python app.py --test "Hello, my name is John Doe and I'm having issues with my billing."
```

### API Endpoints

#### POST /classify-email

Classifies an email and masks PII.

**Request Body:**
```json
{
  "email": "Hello, my name is John Doe, and my email is johndoe@example.com. I'm having issues with my billing."
}
```

**Response:**
```json
{
  "input_email_body": "Hello, my name is John Doe, and my email is johndoe@example.com. I'm having issues with my billing.",
  "list_of_masked_entities": [
    {
      "position": [18, 26],
      "classification": "full_name",
      "entity": "John Doe"
    },
    {
      "position": [42, 61],
      "classification": "email",
      "entity": "johndoe@example.com"
    }
  ],
  "masked_email": "Hello, my name is [full_name], and my email is [email]. I'm having issues with my billing.",
  "category_of_the_email": "Billing Issues"
}
```

#### GET /health

Health check endpoint to verify the API is running.

**Response:**
```json
{
  "status": "healthy"
}
```

## Project Structure

- `app.py`: Main application file
- `models.py`: Model training and classification functions
- `utils.py`: Utility functions for PII masking
- `api.py`: FastAPI implementation for the web service
- `requirements.txt`: Dependencies
- `data/`: Folder for datasets

## Extending the System

### Adding New PII Types

To add new PII types, modify the `patterns` dictionary in the `PiiMasker` class in `utils.py`.

### Using Different Classification Models

To use a different classification model, modify the `pipeline` in the `EmailClassifier` class in `models.py`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.