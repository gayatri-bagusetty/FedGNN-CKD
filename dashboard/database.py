import mysql.connector
import streamlit as st

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