"""
AI Image Detector Backend - Single File FastAPI Server
Integrates with TruthScan API for AI image detection
"""

import os
import uuid
import logging
import json
import time
import random
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from supabase import create_client, Client as SupabaseClient
import requests
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    logger.error("SUPABASE_URL or SUPABASE_SERVICE_KEY not found in environment variables")
    raise ValueError("Supabase credentials are required")

app = FastAPI(title="AI Image Detector Backend", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Supabase client
supabase: SupabaseClient = None


# ============================================================================
# TRUTHSCAN API INTEGRATION CLASS
# ============================================================================

class TruthScanAPI:
    """
    Direct API integration with TruthScan - with retry logic and rate limiting
    """
    
    def __init__(self):
        self.base_url = "https://truthscan.com/api"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Origin': 'https://truthscan.com',
            'Referer': 'https://truthscan.com/ai-image-detector',
            'Sec-Ch-Ua': '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"Windows"',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin'
        })
        self.max_retries = 3
        self.retry_delay = 2
    
    def _retry_request(self, func, *args, **kwargs) -> Dict[str, Any]:
        """Retry a request with exponential backoff"""
        for attempt in range(self.max_retries):
            try:
                result = func(*args, **kwargs)
                if result.get('success') or attempt == self.max_retries - 1:
                    return result
                
                wait_time = self.retry_delay * (2 ** attempt) + random.uniform(0.5, 2.0)
                logger.info(f"Attempt {attempt + 1} failed, retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return {"success": False, "error": str(e)}
                
                wait_time = self.retry_delay * (2 ** attempt) + random.uniform(0.5, 2.0)
                logger.warning(f"Exception on attempt {attempt + 1}: {e}, retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
        
        return {"success": False, "error": "Max retries exceeded"}
    
    def check_image(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze an image for AI generation using direct API calls
        """
        if not Path(image_path).exists():
            return {
                "success": False,
                "error": f"Image not found: {image_path}"
            }
        
        try:
            # Step 1: Upload the image (with retry)
            logger.info(f"Uploading {image_path}")
            upload_result = self._retry_request(self._upload_image, image_path)
            
            if not upload_result.get('success'):
                logger.error(f"Upload failed: {upload_result}")
                return {
                    "success": False,
                    "error": f"Image upload failed: {upload_result.get('error', 'Unknown error')}"
                }
            
            logger.info("Upload successful")
            
            image_url = upload_result['imageUrl']
            r2_file_path = upload_result['r2FilePath']
            
            # Get file info
            file_name = Path(image_path).name
            file_size = os.path.getsize(image_path)
            file_ext = Path(image_path).suffix.lower()
            
            if file_ext in ['.png']:
                file_type = 'image/png'
            elif file_ext in ['.jpg', '.jpeg']:
                file_type = 'image/jpeg'
            else:
                file_type = 'image/jpeg'
            
            # Add delay between requests
            time.sleep(1 + random.uniform(0.5, 1.5))
            
            # Step 2: Detect AI (with retry)
            logger.info("Running AI detection")
            detection_result = self._retry_request(
                self._ai_detect, image_url, r2_file_path, file_name, file_size, file_type
            )
            
            if not detection_result.get('success'):
                logger.error(f"Detection failed: {detection_result}")
                return {
                    "success": False,
                    "error": f"AI detection failed: {detection_result.get('error', 'Unknown error')}"
                }
            
            logger.info("Detection successful")
            
            # Add delay between requests
            time.sleep(1 + random.uniform(0.5, 1.5))
            
            # Step 3: Moderate content
            logger.info("Running moderation check")
            moderation_result = self._moderate_content(image_url)
            
            # Add delay before final analysis
            time.sleep(1 + random.uniform(0.5, 1.5))
            
            # Step 4: Get detailed analysis (optional)
            logger.info("Running detailed analysis")
            analysis_result = ""
            try:
                result = self._ai_image_analysis(
                    image_url,
                    detection_result.get('score', 0),
                    detection_result.get('details', {}).get('result_details', {}).get('final_result', ''),
                    detection_result.get('detectionResultId', ''),
                    detection_result.get('details', {}).get('result_details', {})
                )
                if result and isinstance(result, dict):
                    analysis_result = result.get('analysis', '')
            except Exception as e:
                logger.warning(f"Analysis endpoint failed (non-critical): {e}")
            
            logger.info("Analysis complete")
            
            return self._format_results(detection_result, analysis_result, moderation_result)
            
        except Exception as e:
            logger.error(f"Exception in check_image: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    def _upload_image(self, image_path: str) -> Dict[str, Any]:
        """Step 1: Upload image"""
        url = f"{self.base_url}/upload-image"
        
        try:
            file_name = Path(image_path).name
            file_ext = Path(image_path).suffix.lower()
            
            if file_ext == '.png':
                mime_type = 'image/png'
            elif file_ext in ['.jpg', '.jpeg']:
                mime_type = 'image/jpeg'
            elif file_ext == '.gif':
                mime_type = 'image/gif'
            elif file_ext == '.webp':
                mime_type = 'image/webp'
            else:
                mime_type = 'image/jpeg'
            
            logger.info(f"Uploading {file_name} as {mime_type}")
            
            with open(image_path, 'rb') as f:
                files = {'file': (file_name, f, mime_type)}
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': '*/*',
                    'Origin': 'https://truthscan.com',
                    'Referer': 'https://truthscan.com/ai-image-detector'
                }
                
                response = requests.post(
                    url,
                    files=files,
                    headers=headers,
                    timeout=120
                )
            
            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"{response.status_code} {response.reason}"
                }
            
            return response.json()
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Upload error: {str(e)}"
            }
    
    def _ai_detect(self, image_url: str, r2_file_path: str, file_name: str, 
                   file_size: int, file_type: str) -> Dict[str, Any]:
        """Step 2: Run AI detection"""
        url = f"{self.base_url}/ai-detect-image"
        
        try:
            data = {
                'imageUrl': image_url,
                'fileName': file_name,
                'fileSize': str(file_size),
                'fileType': file_type,
                'r2FilePath': r2_file_path
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': '*/*',
                'Origin': 'https://truthscan.com',
                'Referer': 'https://truthscan.com/ai-image-detector',
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            response = requests.post(
                url,
                data=data,
                headers=headers,
                timeout=120
            )
            
            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"{response.status_code} {response.reason}"
                }
            
            return response.json()
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Detection error: {str(e)}"
            }
    
    def _moderate_content(self, image_url: str) -> Dict[str, Any]:
        """Step 3: Content moderation check"""
        url = f"{self.base_url}/moderate-content"
        
        try:
            data = {'imageUrl': image_url}
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': '*/*',
                'Origin': 'https://truthscan.com',
                'Referer': 'https://truthscan.com/ai-image-detector'
            }
            
            response = requests.post(
                url,
                data=data,
                headers=headers,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except:
            return {"safe": True}
    
    def _ai_image_analysis(self, image_url: str, ai_score: float, final_result: str,
                          detection_result_id: str, result_details: Dict[str, Any]) -> Dict[str, Any]:
        """Step 4: Get detailed AI analysis"""
        url = f"{self.base_url}/ai-image-analysis"
        
        try:
            data = {
                'imageUrl': image_url,
                'aiScore': str(ai_score),
                'finalResult': final_result,
                'detectionResultId': detection_result_id,
                'resultDetails': json.dumps(result_details)
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': '*/*',
                'Origin': 'https://truthscan.com',
                'Referer': 'https://truthscan.com/ai-image-detector'
            }
            
            response = requests.post(
                url,
                data=data,
                headers=headers,
                timeout=90
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Analysis error (non-critical): {e}")
            return {}
    
    def _format_results(self, detection: Dict[str, Any], analysis: str, 
                       moderation: Dict[str, Any]) -> Dict[str, Any]:
        """Format results to match expected output format"""
        ai_percentage = detection.get('score', 0)
        is_ai = detection.get('isAI', False)
        confidence = detection.get('confidence', 'Unknown')
        
        result_details = detection.get('details', {}).get('result_details', {})
        
        return {
            "success": True,
            "verdict": result_details.get('final_result', 'AI Generated' if is_ai else 'Real'),
            "ai_percentage": ai_percentage,
            "human_percentage": 100 - ai_percentage,
            "confidence": confidence,
            "is_ai_generated": is_ai,
            "heatmap_url": result_details.get('heatmap_url'),
            "analysis": analysis if isinstance(analysis, str) else "",
            "metadata": result_details.get('metadata', []),
            "detection_step": result_details.get('detection_step')
        }


# ============================================================================
# FASTAPI ROUTES
# ============================================================================

@app.on_event("startup")
async def startup_event():
    global supabase
    try:
        logger.info("Initializing Supabase client...")
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        logger.info("Supabase client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "supabase_ready": supabase is not None
    }

@app.post("/api/register-device")
async def register_device(request: dict):
    """Register device information"""
    try:
        device_id = request.get('device_id')
        logger.info(f"Registering device: {device_id}")
        
        return JSONResponse({
            "success": True,
            "message": "Device registered successfully"
        })
    except Exception as e:
        logger.error(f"Error registering device: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.post("/api/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    x_device_id: Optional[str] = Header(None),
    x_request_hash: Optional[str] = Header(None)
):
    """
    Analyze image for AI generation using TruthScan API
    This endpoint is called by the Flutter app
    """
    temp_image_path = None
    
    try:
        logger.info(f"Received analysis request from device: {x_device_id}")
        logger.info(f"Request hash: {x_request_hash}")
        logger.info(f"File: {file.filename}, Content-Type: {file.content_type}")
        
        # Validate file type
        valid_content_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        
        content_type = file.content_type or ""
        filename = file.filename or ""
        
        is_valid_content_type = any(content_type.startswith(ct) for ct in ['image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/'])
        is_valid_extension = any(filename.lower().endswith(ext) for ext in valid_extensions)
        
        if not (is_valid_content_type or is_valid_extension):
            logger.warning(f"Invalid file - Content-Type: {content_type}, Filename: {filename}")
            raise HTTPException(status_code=400, detail="File must be an image (jpg, png, webp)")
        
        # Create temp directory
        temp_dir = Path("/tmp")
        temp_dir.mkdir(exist_ok=True)
        
        # Save uploaded image temporarily
        image_id = str(uuid.uuid4())
        file_extension = Path(filename).suffix or '.jpg'
        temp_image_path = temp_dir / f"{image_id}{file_extension}"
        
        # Save file
        with open(temp_image_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Apply EXIF orientation correction
        from PIL import Image, ImageOps
        try:
            with Image.open(temp_image_path) as img:
                corrected_img = ImageOps.exif_transpose(img)
                if corrected_img is None:
                    corrected_img = img
                corrected_img.save(temp_image_path)
                logger.info("Image orientation corrected using EXIF data")
        except Exception as e:
            logger.warning(f"Failed to correct image orientation: {e}, proceeding with original")
        
        logger.info(f"Image saved to {temp_image_path}")
        
        # Validate file size (10MB limit)
        file_size = temp_image_path.stat().st_size
        if file_size > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")
        
        # Call TruthScan analyzer
        logger.info("Calling TruthScan analyzer...")
        detector = TruthScanAPI()
        result = detector.check_image(str(temp_image_path))
        
        if not result.get('success'):
            raise HTTPException(
                status_code=500,
                detail=f"Analysis failed: {result.get('error', 'Unknown error')}"
            )
        
        logger.info(f"Analysis complete: {result.get('verdict')}")
        
        # Return results in format expected by Flutter app
        return JSONResponse({
            "success": True,
            "result": {
                "verdict": result.get('verdict', 'Unknown'),
                "ai_percentage": result.get('ai_percentage', 0),
                "human_percentage": result.get('human_percentage', 100),
                "confidence": result.get('confidence', 'Unknown'),
                "is_ai_generated": result.get('is_ai_generated', False),
                "analysis": result.get('analysis', '')
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze image: {str(e)}"
        )
    finally:
        # Cleanup temporary file
        if temp_image_path and Path(temp_image_path).exists():
            try:
                Path(temp_image_path).unlink()
                logger.info(f"Cleaned up temp file: {temp_image_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {temp_image_path}: {e}")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception handler caught: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error", "detail": str(exc)}
    )

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=10000,
        timeout_keep_alive=300,
        timeout_graceful_shutdown=30
    )
