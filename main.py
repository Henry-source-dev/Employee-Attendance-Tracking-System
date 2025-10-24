from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import sqlite3
import jwt
import bcrypt
from datetime import datetime, timedelta
import numpy as np
import cv2
from PIL import Image
import io
import torch
from retinaface.pre_trained_models import get_model
import logging
from functools import wraps
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"

# Configuration
class Config:
    DATABASE_PATH = 'attendance.db'
    FACE_MODEL = "resnet50_2020-07-20"
    MAX_IMAGE_SIZE = 1048
    DEVICE = "cpu"  # Change to "cuda" if GPU available
    SIMILARITY_THRESHOLD = 0.5
    MAX_FACES_PER_IMAGE = 10

config = Config()

# Face Recognition System
class FaceRecognitionSystem:
    def __init__(self):
        self.model = None
        self.load_model()
        self.init_database()
    
    def load_model(self):
        """Load the RetinaFace model"""
        try:
            self.model = get_model(
                config.FACE_MODEL,
                max_size=config.MAX_IMAGE_SIZE,
                device=config.DEVICE
            )
            self.model.eval()
            logger.info("RetinaFace model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(config.DATABASE_PATH)
        cursor = conn.cursor()
        
        # Employees table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                role TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Face embeddings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (employee_id) REFERENCES employees (employee_id)
            )
        ''')
        
        # Attendance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT NOT NULL,
                check_in_time TIMESTAMP,
                check_out_time TIMESTAMP,
                check_in_lat REAL,
                check_in_lon REAL,
                check_out_lat REAL,
                check_out_lon REAL,
                check_in_image BLOB,
                check_out_image BLOB,
                date DATE NOT NULL,
                confidence REAL,
                FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def extract_face_features(self, image: np.ndarray) -> List[dict]:
        """Extract face features using RetinaFace"""
        try:
            with torch.no_grad():
                annotations = self.model.predict_jsons(image)
            
            if not annotations or not annotations[0].get("bbox"):
                return []
            
            faces = []
            for annotation in annotations[:config.MAX_FACES_PER_IMAGE]:
                bbox = annotation["bbox"]
                landmarks = annotation.get("landmarks", [])
                
                # Extract face region
                x1, y1, x2, y2 = map(int, bbox)
                face_region = image[y1:y2, x1:x2]
                
                # Generate face embedding
                face_embedding = self._generate_face_embedding(face_region)
                
                faces.append({
                    "bbox": bbox,
                    "landmarks": landmarks,
                    "embedding": face_embedding,
                    "face_region": face_region
                })
            
            return faces
        except Exception as e:
            logger.error(f"Failed to extract face features: {str(e)}")
            return []
    
    def _generate_face_embedding(self, face_region: np.ndarray) -> np.ndarray:
        """Generate face embedding using patch-based features"""
        try:
            if face_region.size == 0:
                return np.zeros(512, dtype=np.float32)
            
            # Resize face to standard size
            face_resized = cv2.resize(face_region, (112, 112))
            
            # Convert to grayscale and normalize
            if len(face_resized.shape) == 3:
                face_gray = cv2.cvtColor(face_resized, cv2.COLOR_RGB2GRAY)
            else:
                face_gray = face_resized
            
            face_normalized = face_gray.astype(np.float32) / 255.0
            
            # Create embedding using image patches
            embedding = []
            patch_size = 8
            for i in range(0, 112, patch_size):
                for j in range(0, 112, patch_size):
                    patch = face_normalized[i:i+patch_size, j:j+patch_size]
                    embedding.extend([
                        np.mean(patch),
                        np.std(patch),
                        np.min(patch),
                        np.max(patch)
                    ])
            
            return np.array(embedding[:512], dtype=np.float32)
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            return np.zeros(512, dtype=np.float32)
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            if embedding1.shape != embedding2.shape:
                logger.error(f"Embedding shape mismatch: {embedding1.shape} vs {embedding2.shape}")
                return 0.0
            
            emb1 = embedding1.astype(np.float32)
            emb2 = embedding2.astype(np.float32)
            
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = float(np.dot(emb1, emb2) / (norm1 * norm2))
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {str(e)}")
            return 0.0
    
    def register_employee_face(self, employee_id: str, image_bytes: bytes) -> dict:
        """Register employee face from image bytes"""
        try:
            # Convert bytes to PIL Image
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_array = np.array(pil_image)
            
            # Extract face features
            faces = self.extract_face_features(image_array)
            
            if not faces:
                return {"success": False, "message": "No face detected in image"}
            
            if len(faces) > 1:
                return {"success": False, "message": "Multiple faces detected. Please use image with single face"}
            
            face = faces[0]
            embedding = face["embedding"].astype(np.float32)
            embedding_blob = embedding.tobytes()
            
            # Store in database
            conn = sqlite3.connect(config.DATABASE_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO face_embeddings (employee_id, embedding)
                VALUES (?, ?)
            ''', (employee_id, embedding_blob))
            
            conn.commit()
            conn.close()
            
            return {"success": True, "message": "Face registered successfully"}
        except Exception as e:
            logger.error(f"Failed to register face: {str(e)}")
            return {"success": False, "message": str(e)}
    
    def recognize_employee(self, image_bytes: bytes) -> dict:
        """Recognize employee from image bytes"""
        try:
            # Convert bytes to PIL Image
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_array = np.array(pil_image)
            
            # Extract face features
            faces = self.extract_face_features(image_array)
            
            if not faces:
                return {"success": False, "message": "No face detected"}
            
            # Get all registered embeddings
            conn = sqlite3.connect(config.DATABASE_PATH)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT fe.employee_id, fe.embedding, e.name
                FROM face_embeddings fe
                JOIN employees e ON fe.employee_id = e.employee_id
            ''')
            registered_faces = cursor.fetchall()
            conn.close()
            
            if not registered_faces:
                return {"success": False, "message": "No registered employees found"}
            
            best_matches = []
            for face in faces:
                query_embedding = face["embedding"]
                best_similarity = 0
                best_match = None
                
                for emp_id, embedding_blob, name in registered_faces:
                    stored_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                    similarity = self._calculate_similarity(query_embedding, stored_embedding)
                    
                    if similarity > best_similarity and similarity > config.SIMILARITY_THRESHOLD:
                        best_similarity = similarity
                        best_match = {
                            "employee_id": emp_id,
                            "name": name,
                            "confidence": float(similarity),
                            "bbox": face["bbox"]
                        }
                
                if best_match:
                    best_matches.append(best_match)
            
            if not best_matches:
                return {"success": False, "message": "No matching employee found"}
            
            return {
                "success": True,
                "matches": best_matches,
                "total_faces": len(faces),
                "recognized_faces": len(best_matches)
            }
        except Exception as e:
            logger.error(f"Failed to recognize employee: {str(e)}")
            return {"success": False, "message": str(e)}

# Initialize system
face_system = FaceRecognitionSystem()

# Helper functions
def get_db():
    conn = sqlite3.connect(config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def create_token(employee_id: str, role: str):
    payload = {
        "employee_id": employee_id,
        "role": role,
        "exp": datetime.utcnow() + timedelta(days=7)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Pydantic models
class Login(BaseModel):
    email: str
    password: str

# API Endpoints
@app.get("/")
def root():
    return {"message": "Employee Attendance System API", "version": "1.0"}

@app.post("/api/auth/login")
def login(credentials: Login):
    conn = get_db()
    c = conn.cursor()
    
    c.execute("SELECT * FROM employees WHERE email = ?", (credentials.email,))
    employee = c.fetchone()
    conn.close()
    
    if not employee:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if not bcrypt.checkpw(credentials.password.encode('utf-8'), employee['password']):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_token(employee['employee_id'], employee['role'])
    
    return {
        "token": token,
        "employee": {
            "id": employee['employee_id'],
            "name": employee['name'],
            "email": employee['email'],
            "role": employee['role']
        }
    }

@app.post("/api/admin/register-employee")
async def register_employee(
    employee_id: str = Form(...),
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form("employee"),
    face_image: UploadFile = File(...),
    token_data: dict = Depends(verify_token)
):
    if token_data['role'] != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Read face image
    image_bytes = await face_image.read()
    
    # Register face first
    face_result = face_system.register_employee_face(employee_id, image_bytes)
    
    if not face_result["success"]:
        raise HTTPException(status_code=400, detail=face_result["message"])
    
    # Hash password
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute(
            "INSERT INTO employees (employee_id, name, email, password, role) VALUES (?, ?, ?, ?, ?)",
            (employee_id, name, email, hashed_password, role)
        )
        conn.commit()
        conn.close()
        
        return {"message": "Employee registered successfully", "employee_id": employee_id}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Employee ID or email already exists")

@app.post("/api/attendance/check-in")
async def check_in(
    latitude: float = Form(...),
    longitude: float = Form(...),
    face_image: UploadFile = File(...),
    token_data: dict = Depends(verify_token)
):
    employee_id = token_data['employee_id']
    
    # Read face image
    image_bytes = await face_image.read()
    
    # Recognize employee
    recognition_result = face_system.recognize_employee(image_bytes)
    
    if not recognition_result["success"] or not recognition_result.get("matches"):
        raise HTTPException(status_code=401, detail="Face verification failed")
    
    # Find matching employee
    best_match = None
    for match in recognition_result["matches"]:
        if match["employee_id"] == employee_id:
            best_match = match
            break
    
    if not best_match:
        raise HTTPException(status_code=401, detail="Face does not match registered employee")
    
    # Check if already checked in today
    conn = get_db()
    c = conn.cursor()
    today = datetime.now().date()
    
    c.execute(
        "SELECT id FROM attendance WHERE employee_id = ? AND date = ? AND check_in_time IS NOT NULL",
        (employee_id, today)
    )
    existing = c.fetchone()
    
    if existing:
        conn.close()
        raise HTTPException(status_code=400, detail="Already checked in today")
    
    # Create check-in record
    check_in_time = datetime.now()
    c.execute(
        "INSERT INTO attendance (employee_id, check_in_time, check_in_lat, check_in_lon, check_in_image, date, confidence) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (employee_id, check_in_time, latitude, longitude, image_bytes, today, best_match["confidence"])
    )
    conn.commit()
    conn.close()
    
    return {
        "message": "Checked in successfully",
        "time": check_in_time.isoformat(),
        "location": {"latitude": latitude, "longitude": longitude},
        "confidence": best_match["confidence"]
    }

@app.post("/api/attendance/check-out")
async def check_out(
    latitude: float = Form(...),
    longitude: float = Form(...),
    face_image: UploadFile = File(...),
    token_data: dict = Depends(verify_token)
):
    employee_id = token_data['employee_id']
    
    # Read face image
    image_bytes = await face_image.read()
    
    # Recognize employee
    recognition_result = face_system.recognize_employee(image_bytes)
    
    if not recognition_result["success"] or not recognition_result.get("matches"):
        raise HTTPException(status_code=401, detail="Face verification failed")
    
    # Find matching employee
    best_match = None
    for match in recognition_result["matches"]:
        if match["employee_id"] == employee_id:
            best_match = match
            break
    
    if not best_match:
        raise HTTPException(status_code=401, detail="Face does not match registered employee")
    
    # Find today's check-in record
    conn = get_db()
    c = conn.cursor()
    today = datetime.now().date()
    
    c.execute(
        "SELECT * FROM attendance WHERE employee_id = ? AND date = ? AND check_in_time IS NOT NULL AND check_out_time IS NULL",
        (employee_id, today)
    )
    record = c.fetchone()
    
    if not record:
        conn.close()
        raise HTTPException(status_code=400, detail="No active check-in found for today")
    
    # Update with check-out
    check_out_time = datetime.now()
    c.execute(
        "UPDATE attendance SET check_out_time = ?, check_out_lat = ?, check_out_lon = ?, check_out_image = ? WHERE id = ?",
        (check_out_time, latitude, longitude, image_bytes, record['id'])
    )
    conn.commit()
    conn.close()
    
    return {
        "message": "Checked out successfully",
        "time": check_out_time.isoformat(),
        "location": {"latitude": latitude, "longitude": longitude},
        "confidence": best_match["confidence"]
    }

@app.get("/api/admin/attendance")
def get_attendance(
    date: Optional[str] = None,
    employee_id: Optional[str] = None,
    token_data: dict = Depends(verify_token)
):
    if token_data['role'] != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    
    conn = get_db()
    c = conn.cursor()
    
    query = """
        SELECT a.*, e.name, e.email
        FROM attendance a
        JOIN employees e ON a.employee_id = e.employee_id
        WHERE 1=1
    """
    params = []
    
    if date:
        query += " AND a.date = ?"
        params.append(date)
    
    if employee_id:
        query += " AND a.employee_id = ?"
        params.append(employee_id)
    
    query += " ORDER BY a.date DESC, a.check_in_time DESC"
    
    c.execute(query, params)
    records = c.fetchall()
    conn.close()
    
    result = []
    for record in records:
        result.append({
            "id": record['id'],
            "employee_id": record['employee_id'],
            "name": record['name'],
            "email": record['email'],
            "date": record['date'],
            "check_in_time": record['check_in_time'],
            "check_out_time": record['check_out_time'],
            "check_in_location": {
                "latitude": record['check_in_lat'],
                "longitude": record['check_in_lon']
            } if record['check_in_lat'] else None,
            "check_out_location": {
                "latitude": record['check_out_lat'],
                "longitude": record['check_out_lon']
            } if record['check_out_lat'] else None,
            "confidence": record['confidence']
        })
    
    return {"records": result}

@app.get("/api/admin/employees")
def get_employees(token_data: dict = Depends(verify_token)):
    if token_data['role'] != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT employee_id, name, email, role, created_at FROM employees")
    employees = c.fetchall()
    conn.close()
    
    return {
        "employees": [
            {
                "employee_id": e['employee_id'],
                "name": e['name'],
                "email": e['email'],
                "role": e['role'],
                "created_at": e['created_at']
            }
            for e in employees
        ]
    }

@app.get("/api/employee/my-attendance")
def get_my_attendance(token_data: dict = Depends(verify_token)):
    employee_id = token_data['employee_id']
    
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "SELECT * FROM attendance WHERE employee_id = ? ORDER BY date DESC LIMIT 30",
        (employee_id,)
    )
    records = c.fetchall()
    conn.close()
    
    result = []
    for record in records:
        result.append({
            "id": record['id'],
            "date": record['date'],
            "check_in_time": record['check_in_time'],
            "check_out_time": record['check_out_time'],
            "check_in_location": {
                "latitude": record['check_in_lat'],
                "longitude": record['check_in_lon']
            } if record['check_in_lat'] else None,
            "check_out_location": {
                "latitude": record['check_out_lat'],
                "longitude": record['check_out_lon']
            } if record['check_out_lat'] else None,
            "confidence": record['confidence']
        })
    
    return {"records": result}

@app.get("/api/employee/status")
def get_status(token_data: dict = Depends(verify_token)):
    employee_id = token_data['employee_id']
    today = datetime.now().date()
    
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "SELECT * FROM attendance WHERE employee_id = ? AND date = ?",
        (employee_id, today)
    )
    record = c.fetchone()
    conn.close()
    
    return {
        "checked_in": record is not None and record['check_in_time'] is not None,
        "checked_out": record is not None and record['check_out_time'] is not None,
        "check_in_time": record['check_in_time'] if record else None,
        "check_out_time": record['check_out_time'] if record else None
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)