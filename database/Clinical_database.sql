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