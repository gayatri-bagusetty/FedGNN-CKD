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
        
def get_classification_stats():
    conn = get_db_connection()
    if not conn: return {"CKD": 0, "Non-CKD": 0}
    try:
        cursor = conn.cursor(dictionary=True)
        # Counts how many 'CKD' and 'Non-CKD' entries exist
        cursor.execute("SELECT Result, COUNT(*) as count FROM patients GROUP BY Result")
        rows = cursor.fetchall()
        
        stats = {"CKD": 0, "Non-CKD": 0}
        for row in rows:
            if row['Result'] in stats:
                stats[row['Result']] = row['count']
        return stats
    except:
        return {"CKD": 0, "Non-CKD": 0}
    finally:
        conn.close()
        

def get_federated_state():
    conn = get_db_connection()
    if not conn: return {"round": 0, "hospitals": 0}
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT meta_key, meta_value FROM system_metadata WHERE meta_key IN ('fed_round', 'active_hospitals')")
        rows = cursor.fetchall()
        state = {"round": 0, "hospitals": 0}
        for row in rows:
            if row['meta_key'] == 'fed_round': state['round'] = int(row['meta_value'])
            if row['meta_key'] == 'active_hospitals': state['hospitals'] = int(row['meta_value'])
        return state
    except:
        return {"round": 0, "hospitals": 0}
    finally:
        conn.close()

def update_federated_state(fed_round, hospitals_count):
    conn = get_db_connection()
    if not conn: return
    try:
        cursor = conn.cursor()
        cursor.execute("UPDATE system_metadata SET meta_value = %s WHERE meta_key = 'fed_round'", (str(fed_round),))
        cursor.execute("UPDATE system_metadata SET meta_value = %s WHERE meta_key = 'active_hospitals'", (str(hospitals_count),))
        conn.commit()
    finally:
        conn.close()