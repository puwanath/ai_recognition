from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image
import io
import re
import logging
import gc

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Determine device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    logger.info("Using MPS device")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    logger.info("Using CUDA device")
else:
    DEVICE = torch.device("cpu")
    logger.info("Using CPU device")

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model and processor globally
try:
    logger.info("Loading model and processor...")
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        # trust_remote_code=True,
        torch_dtype=torch.float16 if DEVICE.type != "cpu" else torch.float32,
        device_map=DEVICE
    ).to(DEVICE)
    
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    logger.info(f"Model and processor loaded successfully on {DEVICE}")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

def clear_memory():
    """Clear memory cache based on device type"""
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

async def process_image(image: UploadFile, prompt: str):
    try:
        clear_memory()
        
        # Read and process the uploaded image
        image_content = await image.read()
        image = Image.open(io.BytesIO(image_content))
        
        # Resize image if too large
        max_size = 800
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.LANCZOS)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Prepare for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Prepare inputs and move to correct device
        inputs = processor(
            images=image,
            text=[text],
            padding=True,
            return_tensors="pt"
        ).to(device=DEVICE, dtype=torch.float16 if DEVICE.type != "cpu" else torch.float32)
        
        # Use torch.no_grad for inference
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]  # Get first element since we're processing single image
        
        clear_memory()
        return output_text
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/read_meter")
async def read_meter(image: UploadFile = File(...)):
    try:
        prompt = "Is this image a utility meter (water, electricity, or gas meter)? If yes, what type is it and what is the running number? Answer in a structured way."
        result = await process_image(image, prompt)
        
        # Process result
        is_meter = "meter" in result.lower()
        meter_type = None
        reading = "0000"
        
        if is_meter:
            if "water" in result.lower():
                meter_type = "water"
            elif "electricity" in result.lower() or "electric" in result.lower():
                meter_type = "electricity"
            elif "gas" in result.lower():
                meter_type = "gas"
                
            # Extract numbers
            numbers = re.findall(r'\d+', result)
            if numbers:
                reading = numbers[0]
        
        return {
            "status": is_meter,
            "metertype": meter_type,
            "readingnum": reading
        }
    
    except Exception as e:
        logger.error(f"Error in read_meter: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_football_field")
async def analyze_football_field(image: UploadFile = File(...)):
    try:
        prompt = "Analyze this image. Is it a football field? Are there people playing football? How many people can you see? Describe the activity briefly."
        result = await process_image(image, prompt)
        
        # Process result
        has_people = "person" in result.lower() or "people" in result.lower()
        playing_football = "playing" in result.lower() and "football" in result.lower()
        
        # Extract number of people
        numbers = re.findall(r'\d+', result)
        people_count = 0
        for num in numbers:
            if int(num) < 100:  # Reasonable number of people
                people_count = int(num)
                break
        
        return {
            "status": has_people,
            "playfootball": "yes" if playing_football else "no",
            "desc": result[:100],  # First 100 characters
            "peoplecount": people_count
        }
    
    except Exception as e:
        logger.error(f"Error in analyze_football_field: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/read_receipt")
async def read_receipt(image: UploadFile = File(...)):
    try:
        prompt = "Is this a receipt or tax invoice? If yes, what is the total amount? Extract only the final total amount."
        result = await process_image(image, prompt)
        
        # Process result
        is_receipt = "receipt" in result.lower() or "invoice" in result.lower()
        total_amount = 0
        
        # Extract amount
        amounts = re.findall(r'\d+\.?\d*', result)
        if amounts:
            try:
                total_amount = float(amounts[-1])  # Usually the last number is the total
            except ValueError:
                pass
        
        return {
            "status": is_receipt,
            "totalamount": total_amount
        }
    
    except Exception as e:
        logger.error(f"Error in read_receipt: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "device_type": DEVICE.type,
        "model_name": "Qwen2.5-VL-3B-Instruct"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)