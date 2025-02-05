# Image Analysis API with Qwen-VL

This project implements a FastAPI-based REST API that uses the Qwen2.5-VL (Visual Language) model for various image analysis tasks. The API provides endpoints for reading utility meters, analyzing football field activities, and extracting information from receipts.

## Features

- **Utility Meter Reading**: Identifies and reads values from water, electricity, and gas meters
- **Football Field Analysis**: Detects people playing football and provides activity descriptions
- **Receipt Analysis**: Extracts total amounts from receipts and invoices
- **Multi-platform Support**: Works on CPU, CUDA (NVIDIA GPUs), and MPS (Apple Silicon)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- FastAPI
- Transformers library
- 8GB+ RAM recommended
- CUDA-compatible GPU (optional)
- Apple Silicon Mac (optional, for MPS acceleration)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/puwanath/ai_recognition
cd ai_recognition
```

2. Install dependencies:
```bash
pip install fastapi uvicorn python-multipart torch transformers pillow
```

## Configuration

The application automatically detects and uses the best available hardware:
- Apple Silicon Macs: MPS (Metal Performance Shaders)
- NVIDIA GPUs: CUDA
- Other systems: CPU

## Running the Application

Start the server:
```bash
python app.py
```
Or use uvicorn directly:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Health Check
```http
GET /health
```
Returns the service status and device information.

### Read Utility Meter
```http
POST /read_meter
```
Analyzes utility meter images and returns:
```json
{
    "status": true/false,
    "metertype": "water/electricity/gas",
    "readingnum": "0000"
}
```

### Analyze Football Field
```http
POST /analyze_football_field
```
Analyzes football field activities and returns:
```json
{
    "status": true/false,
    "playfootball": "yes/no",
    "desc": "Activity description",
    "peoplecount": 0
}
```

### Read Receipt
```http
POST /read_receipt
```
Extracts total amount from receipts and returns:
```json
{
    "status": true/false,
    "totalamount": 0
}
```

## Usage Examples

Using curl:
```bash
# Read meter
curl -X POST -F "image=@/path/to/meter.jpg" http://localhost:8000/read_meter

# Analyze football field
curl -X POST -F "image=@/path/to/field.jpg" http://localhost:8000/analyze_football_field

# Read receipt
curl -X POST -F "image=@/path/to/receipt.jpg" http://localhost:8000/read_receipt
```

Using Python requests:
```python
import requests

files = {"image": ("image.jpg", open("path/to/image.jpg", "rb"))}
response = requests.post("http://localhost:8000/read_meter", files=files)
print(response.json())
```

## Memory Management

The application includes several memory optimization features:
- Automatic image resizing for large images
- Memory cache clearing after each inference
- Device-specific optimizations
- Garbage collection

## Error Handling

The API includes comprehensive error handling and logging:
- Input validation
- Processing errors
- Model inference errors
- Memory-related errors

## Limitations

- Image size: Large images are automatically resized to max dimension of 800px
- Memory usage: Depends on available system resources
- Processing time: Varies based on hardware and image complexity

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Acknowledgments

- Qwen team for the Qwen2.5-VL model
- FastAPI team for the web framework
- Hugging Face team for the transformers library
