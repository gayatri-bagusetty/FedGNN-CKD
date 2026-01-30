-- 1. Create the database
CREATE DATABASE clinical_db;

-- 2. Use the database
USE clinical_db;

-- 3. Create the users table
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    role VARCHAR(20) NOT NULL, -- 'Doctor' or 'Admin'
    user_id VARCHAR(50) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL
);

-- 4. Insert sample data for testing
INSERT INTO users (role, user_id, password) VALUES 
('Doctor', '41568', 'doctor123'),
('Admin', '56987', 'admin123');

ALTER TABLE users ADD COLUMN full_name VARCHAR(100);

-- Update your existing doctor record
UPDATE users SET full_name = 'Dr. John Smith' WHERE user_id = '41568';

-- Update your existing admin record
UPDATE users SET full_name = 'Kavin Smith' WHERE user_id = '56987';

INSERT INTO users (user_id, password, full_name, role)
VALUES ('52890', 'securepass', 'Dr. Sarah Miller', 'Doctor');

SELECT * FROM users;

-- Create the patients table
CREATE TABLE IF NOT EXISTS patients (
    id INT AUTO_INCREMENT PRIMARY KEY,
    TimeOfEntry DATETIME NOT NULL,
    Result VARCHAR(50) NOT NULL,
    age INT,
    bp INT,
    sg FLOAT,
    al INT,
    su INT,
    rbc VARCHAR(20),
    pc VARCHAR(20),
    pcc VARCHAR(20),
    ba VARCHAR(20),
    bgr INT,
    bu INT,
    sc FLOAT,
    sod FLOAT,
    pot FLOAT,
    hemo FLOAT,
    pcv INT,
    wbcc INT,
    rbcc FLOAT,
    htn VARCHAR(10),
    dm VARCHAR(10),
    cad VARCHAR(10),
    appet VARCHAR(20),
    pe VARCHAR(10),
    ane VARCHAR(10)
);