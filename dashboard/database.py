import mysql.connector
import streamlit as st
from datetime import datetime

# --- Connection Helper ---
def get_db_connection():
    try:
        return mysql.connector.connect(
            host="localhost",
            user="root",
            password="1997",
            database="clinical_db"
        )
    except Exception as e:
        st.error(f"Database Connection Error: {e}")
        return None

# --- Functions ---
def verify_user(role, user_id, password):
    conn = get_db_connection()
    if not conn: return None
    try:
        cursor = conn.cursor(dictionary=True) 
        query = "SELECT * FROM users WHERE role = %s AND user_id = %s AND password = %s"
        cursor.execute(query, (role, user_id, password))
        user = cursor.fetchone() 
        return user 
    except Exception as e:
        return None
    finally:
        conn.close()

def get_total_patients():
    conn = get_db_connection()
    if not conn: return 0
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM patients")
        res = cursor.fetchone()
        return res[0] if res else 0
    except:
        return 0
    finally:
        conn.close()

def fetch_all_patients():
    conn = get_db_connection()
    if not conn: return []
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM patients ORDER BY TimeOfEntry DESC")
        return cursor.fetchall()
    except Exception as e:
        st.error(f"Database Error: {e}")
        return []
    finally:
        conn.close()

def save_patient_data(raw_patient_data, result):
    conn = get_db_connection()
    if not conn: return
    try:
        cursor = conn.cursor()
        data_to_insert = {
            "TimeOfEntry": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Result": result,
            **raw_patient_data
        }
        columns = ", ".join(data_to_insert.keys())
        placeholders = ", ".join(["%s"] * len(data_to_insert))
        values = tuple(data_to_insert.values())
        insert_query = f"INSERT INTO patients ({columns}) VALUES ({placeholders})"
        cursor.execute(insert_query, values)
        conn.commit()
    except Exception as e:
        st.error(f"Database Error: {e}")
    finally:
        conn.close()
        

def get_total_users():
    conn = get_db_connection()
    if not conn: return 0
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        res = cursor.fetchone()
        return res[0] if res else 0
    except:
        return 0
    finally:
        conn.close()