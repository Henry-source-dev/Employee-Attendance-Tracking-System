import sqlite3
import bcrypt
from datetime import datetime

def create_admin():
    """Create initial admin user for the system"""
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    
    # Admin credentials
    admin_id = "ADMIN001"
    name = "System Administrator"
    email = "admin@company.com"
    password = "admin123"  # Change this in production!
    role = "admin"
    
    # Hash password
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    try:
        c.execute(
            "INSERT INTO employees (employee_id, name, email, password, role, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (admin_id, name, email, hashed_password, role, datetime.now())
        )
        conn.commit()
        print("✓ Admin user created successfully!")
        print(f"  Email: {email}")
        print(f"  Password: {password}")
        print(f"  Employee ID: {admin_id}")
        print("\n⚠️  IMPORTANT: Change the admin password after first login!")
    except sqlite3.IntegrityError:
        print("✗ Admin user already exists!")
    finally:
        conn.close()

if __name__ == "__main__":
    create_admin()