import mysql.connector
import streamlit as st
from datetime import datetime

def verify_user(role, user_id, password):
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="1997",
            database="clinical_db"
        )
        cursor = conn.cursor(dictionary=True) 
        query = "SELECT * FROM users WHERE role = %s AND user_id = %s AND password = %s"
        cursor.execute(query, (role, user_id, password))
        user = cursor.fetchone() 
        return user 
    except Exception as e:
        return None
    
# --- New function to save patient data ---
def save_patient_data(raw_patient_data, result):
    """
    raw_patient_data: dict containing all patient info
    result: diagnosis result string (e.g., "CKD" or "Non-CKD")
    """
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="1997",
            database="clinical_db"
        )
        cursor = conn.cursor()

        # Add timestamp and result to the data
        data_to_insert = {
            "TimeOfEntry": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Result": result,
            **raw_patient_data
        }

        # Prepare columns and placeholders
        columns = ", ".join(data_to_insert.keys())
        placeholders = ", ".join(["%s"] * len(data_to_insert))
        values = tuple(data_to_insert.values())

        # Insert query (replace 'patients' with your table name)
        insert_query = f"INSERT INTO patients ({columns}) VALUES ({placeholders})"
        cursor.execute(insert_query, values)
        conn.commit()

        cursor.close()
        conn.close()

    except Exception as e:
        st.error(f"Database Error: {e}")
        
def fetch_all_patients():
    """
    Fetch all patient records from the 'patients' table.
    Returns a list of dictionaries or empty list if none.
    """
    conn = None
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="1997",
            database="clinical_db"
        )
        cursor = conn.cursor(dictionary=True)  # returns rows as dicts
        query = "SELECT * FROM patients ORDER BY TimeOfEntry DESC"
        cursor.execute(query)
        patients = cursor.fetchall()  # list of dicts
        return patients

    except Exception as e:
        st.error(f"Database Error: {e}")
        return []

    finally:
        if conn:
            conn.close()
            
def get_total_patients():
    try:
        # Replace with your actual database connection details
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="1997",
            database="clinical_db"
        )
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM patients")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except Exception as e:
        # Fallback to session state length if DB fails or isn't setup yet
        return len(st.session_state.get('patients', []))