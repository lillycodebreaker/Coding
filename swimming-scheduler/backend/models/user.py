from config.database import db_config
from mysql.connector import Error

class User:
    def __init__(self, id=None, name=None, email=None, phone=None):
        self.id = id
        self.name = name
        self.email = email
        self.phone = phone

    @staticmethod
    def get_all():
        """Get all users"""
        connection = db_config.get_connection()
        if not connection:
            return []
        
        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users ORDER BY name")
            users = cursor.fetchall()
            return users
        except Error as e:
            print(f"Error fetching users: {e}")
            return []
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

    @staticmethod
    def get_by_id(user_id):
        """Get user by ID"""
        connection = db_config.get_connection()
        if not connection:
            return None
        
        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
            user = cursor.fetchone()
            return user
        except Error as e:
            print(f"Error fetching user: {e}")
            return None
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

    @staticmethod
    def create(name, email, phone=None):
        """Create a new user"""
        connection = db_config.get_connection()
        if not connection:
            return None
        
        try:
            cursor = connection.cursor()
            cursor.execute(
                "INSERT INTO users (name, email, phone) VALUES (%s, %s, %s)",
                (name, email, phone)
            )
            user_id = cursor.lastrowid
            return user_id
        except Error as e:
            print(f"Error creating user: {e}")
            return None
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

    @staticmethod
    def update(user_id, name=None, email=None, phone=None):
        """Update user information"""
        connection = db_config.get_connection()
        if not connection:
            return False
        
        try:
            cursor = connection.cursor()
            updates = []
            values = []
            
            if name:
                updates.append("name = %s")
                values.append(name)
            if email:
                updates.append("email = %s")
                values.append(email)
            if phone is not None:
                updates.append("phone = %s")
                values.append(phone)
            
            if updates:
                query = f"UPDATE users SET {', '.join(updates)} WHERE id = %s"
                values.append(user_id)
                cursor.execute(query, values)
                return cursor.rowcount > 0
            return False
        except Error as e:
            print(f"Error updating user: {e}")
            return False
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

    @staticmethod
    def delete(user_id):
        """Delete a user"""
        connection = db_config.get_connection()
        if not connection:
            return False
        
        try:
            cursor = connection.cursor()
            cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
            return cursor.rowcount > 0
        except Error as e:
            print(f"Error deleting user: {e}")
            return False
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()