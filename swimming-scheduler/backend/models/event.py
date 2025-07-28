from config.database import db_config
from mysql.connector import Error
from datetime import datetime

class Event:
    def __init__(self, id=None, title=None, description=None, location=None, 
                 created_by=None, event_date=None, start_time=None, end_time=None, status='planning'):
        self.id = id
        self.title = title
        self.description = description
        self.location = location
        self.created_by = created_by
        self.event_date = event_date
        self.start_time = start_time
        self.end_time = end_time
        self.status = status

    @staticmethod
    def create_with_proposed_dates(title, description, location, created_by, proposed_dates, participant_ids):
        """Create an event with proposed dates and participants"""
        connection = db_config.get_connection()
        if not connection:
            return None
        
        try:
            cursor = connection.cursor()
            
            # Insert event
            cursor.execute(
                """INSERT INTO events (title, description, location, created_by, event_date, status) 
                   VALUES (%s, %s, %s, %s, %s, 'planning')""",
                (title, description, location, created_by, proposed_dates[0]['date'])
            )
            event_id = cursor.lastrowid
            
            # Insert proposed dates
            for date_info in proposed_dates:
                cursor.execute(
                    """INSERT INTO event_proposed_dates (event_id, proposed_date, start_time, end_time) 
                       VALUES (%s, %s, %s, %s)""",
                    (event_id, date_info['date'], date_info.get('start_time'), date_info.get('end_time'))
                )
            
            # Add participants
            for participant_id in participant_ids:
                cursor.execute(
                    "INSERT INTO event_participants (event_id, user_id) VALUES (%s, %s)",
                    (event_id, participant_id)
                )
            
            return event_id
        except Error as e:
            print(f"Error creating event: {e}")
            connection.rollback()
            return None
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

    @staticmethod
    def get_by_id(event_id):
        """Get event details with proposed dates"""
        connection = db_config.get_connection()
        if not connection:
            return None
        
        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute("""
                SELECT 
                    e.id as event_id,
                    e.title,
                    e.description,
                    e.location,
                    e.status,
                    u.name as created_by_name,
                    u.id as created_by_id,
                    epd.id as proposed_date_id,
                    epd.proposed_date,
                    epd.start_time,
                    epd.end_time
                FROM events e
                JOIN users u ON e.created_by = u.id
                LEFT JOIN event_proposed_dates epd ON e.id = epd.event_id
                WHERE e.id = %s
                ORDER BY epd.proposed_date
            """, (event_id,))
            
            rows = cursor.fetchall()
            if not rows:
                return None
            
            # Structure the data
            event = {
                'id': rows[0]['event_id'],
                'title': rows[0]['title'],
                'description': rows[0]['description'],
                'location': rows[0]['location'],
                'status': rows[0]['status'],
                'created_by_name': rows[0]['created_by_name'],
                'created_by_id': rows[0]['created_by_id'],
                'proposed_dates': []
            }
            
            for row in rows:
                if row['proposed_date_id']:
                    event['proposed_dates'].append({
                        'id': row['proposed_date_id'],
                        'date': row['proposed_date'].strftime('%Y-%m-%d'),
                        'start_time': str(row['start_time']) if row['start_time'] else None,
                        'end_time': str(row['end_time']) if row['end_time'] else None
                    })
            
            return event
        except Error as e:
            print(f"Error fetching event: {e}")
            return None
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

    @staticmethod
    def get_events_for_user(user_id):
        """Get events where user is a participant"""
        connection = db_config.get_connection()
        if not connection:
            return []
        
        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute("""
                SELECT 
                    e.id,
                    e.title,
                    e.description,
                    e.location,
                    e.status,
                    u.name as created_by,
                    ep.invitation_status,
                    COUNT(epd.id) as proposed_dates_count
                FROM events e
                JOIN users u ON e.created_by = u.id
                JOIN event_participants ep ON e.id = ep.event_id
                LEFT JOIN event_proposed_dates epd ON e.id = epd.event_id
                WHERE ep.user_id = %s
                GROUP BY e.id
                ORDER BY e.created_at DESC
            """, (user_id,))
            
            events = cursor.fetchall()
            return events
        except Error as e:
            print(f"Error fetching user events: {e}")
            return []
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

    @staticmethod
    def get_events_created_by_user(user_id):
        """Get events created by user"""
        connection = db_config.get_connection()
        if not connection:
            return []
        
        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute("""
                SELECT 
                    e.id,
                    e.title,
                    e.description,
                    e.location,
                    e.status,
                    COUNT(DISTINCT ep.user_id) as participant_count,
                    COUNT(DISTINCT epd.id) as proposed_dates_count
                FROM events e
                LEFT JOIN event_participants ep ON e.id = ep.event_id
                LEFT JOIN event_proposed_dates epd ON e.id = epd.event_id
                WHERE e.created_by = %s
                GROUP BY e.id
                ORDER BY e.created_at DESC
            """, (user_id,))
            
            events = cursor.fetchall()
            return events
        except Error as e:
            print(f"Error fetching created events: {e}")
            return []
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

    @staticmethod
    def get_availability_report(event_id):
        """Get availability report for an event"""
        connection = db_config.get_connection()
        if not connection:
            return []
        
        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute("""
                SELECT 
                    epd.id as proposed_date_id,
                    epd.proposed_date,
                    epd.start_time,
                    epd.end_time,
                    COUNT(ep.user_id) as total_invited,
                    COUNT(CASE WHEN ua.is_available = 1 THEN 1 END) as available_count,
                    GROUP_CONCAT(
                        CASE WHEN ua.is_available = 1 
                        THEN u.name 
                        END SEPARATOR ', '
                    ) as available_users,
                    GROUP_CONCAT(
                        CASE WHEN ua.is_available = 0 OR ua.is_available IS NULL 
                        THEN u.name 
                        END SEPARATOR ', '
                    ) as unavailable_users,
                    ROUND(
                        (COUNT(CASE WHEN ua.is_available = 1 THEN 1 END) * 100.0 / COUNT(ep.user_id)), 
                        1
                    ) as availability_percentage
                FROM event_proposed_dates epd
                JOIN event_participants ep ON epd.event_id = ep.event_id
                JOIN users u ON ep.user_id = u.id
                LEFT JOIN user_availability ua ON epd.id = ua.proposed_date_id AND u.id = ua.user_id
                WHERE epd.event_id = %s
                GROUP BY epd.id, epd.proposed_date, epd.start_time, epd.end_time
                ORDER BY available_count DESC, availability_percentage DESC
            """, (event_id,))
            
            report = cursor.fetchall()
            
            # Format the report
            formatted_report = []
            for row in report:
                formatted_report.append({
                    'proposed_date_id': row['proposed_date_id'],
                    'date': row['proposed_date'].strftime('%Y-%m-%d'),
                    'start_time': str(row['start_time']) if row['start_time'] else None,
                    'end_time': str(row['end_time']) if row['end_time'] else None,
                    'total_invited': row['total_invited'],
                    'available_count': row['available_count'],
                    'available_users': row['available_users'].split(', ') if row['available_users'] else [],
                    'unavailable_users': row['unavailable_users'].split(', ') if row['unavailable_users'] else [],
                    'availability_percentage': float(row['availability_percentage']) if row['availability_percentage'] else 0.0
                })
            
            return formatted_report
        except Error as e:
            print(f"Error fetching availability report: {e}")
            return []
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

    @staticmethod
    def update_user_availability(user_id, event_id, proposed_date_id, is_available, notes=None):
        """Update user availability for a proposed date"""
        connection = db_config.get_connection()
        if not connection:
            return False
        
        try:
            cursor = connection.cursor()
            cursor.execute("""
                INSERT INTO user_availability (user_id, event_id, proposed_date_id, is_available, notes)
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                is_available = VALUES(is_available),
                notes = VALUES(notes),
                response_time = CURRENT_TIMESTAMP
            """, (user_id, event_id, proposed_date_id, is_available, notes))
            
            return True
        except Error as e:
            print(f"Error updating availability: {e}")
            return False
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

    @staticmethod
    def get_user_availability_for_event(user_id, event_id):
        """Get user's availability for all proposed dates of an event"""
        connection = db_config.get_connection()
        if not connection:
            return []
        
        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute("""
                SELECT 
                    epd.id as proposed_date_id,
                    epd.proposed_date,
                    epd.start_time,
                    epd.end_time,
                    COALESCE(ua.is_available, FALSE) as is_available,
                    ua.notes
                FROM event_proposed_dates epd
                LEFT JOIN user_availability ua ON epd.id = ua.proposed_date_id AND ua.user_id = %s
                WHERE epd.event_id = %s
                ORDER BY epd.proposed_date
            """, (user_id, event_id))
            
            availability = cursor.fetchall()
            
            # Format the response
            formatted_availability = []
            for row in availability:
                formatted_availability.append({
                    'proposed_date_id': row['proposed_date_id'],
                    'date': row['proposed_date'].strftime('%Y-%m-%d'),
                    'start_time': str(row['start_time']) if row['start_time'] else None,
                    'end_time': str(row['end_time']) if row['end_time'] else None,
                    'is_available': bool(row['is_available']),
                    'notes': row['notes']
                })
            
            return formatted_availability
        except Error as e:
            print(f"Error fetching user availability: {e}")
            return []
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()