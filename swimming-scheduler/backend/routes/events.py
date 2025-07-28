from flask import Blueprint, request, jsonify
from models.event import Event

events_bp = Blueprint('events', __name__)

@events_bp.route('/events', methods=['POST'])
def create_event():
    """Create a new event with proposed dates"""
    try:
        data = request.get_json()
        
        required_fields = ['title', 'created_by', 'proposed_dates', 'participant_ids']
        for field in required_fields:
            if not data or field not in data:
                return jsonify({
                    'success': False,
                    'error': f'{field} is required'
                }), 400
        
        if not data['proposed_dates'] or len(data['proposed_dates']) == 0:
            return jsonify({
                'success': False,
                'error': 'At least one proposed date is required'
            }), 400
        
        event_id = Event.create_with_proposed_dates(
            title=data['title'],
            description=data.get('description', ''),
            location=data.get('location', ''),
            created_by=data['created_by'],
            proposed_dates=data['proposed_dates'],
            participant_ids=data['participant_ids']
        )
        
        if event_id:
            event = Event.get_by_id(event_id)
            return jsonify({
                'success': True,
                'data': event
            }), 201
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to create event'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@events_bp.route('/events/<int:event_id>', methods=['GET'])
def get_event(event_id):
    """Get event details"""
    try:
        event = Event.get_by_id(event_id)
        if event:
            return jsonify({
                'success': True,
                'data': event
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Event not found'
            }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@events_bp.route('/users/<int:user_id>/events', methods=['GET'])
def get_user_events(user_id):
    """Get events for a user (as participant)"""
    try:
        events = Event.get_events_for_user(user_id)
        return jsonify({
            'success': True,
            'data': events
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@events_bp.route('/users/<int:user_id>/created-events', methods=['GET'])
def get_user_created_events(user_id):
    """Get events created by a user"""
    try:
        events = Event.get_events_created_by_user(user_id)
        return jsonify({
            'success': True,
            'data': events
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@events_bp.route('/events/<int:event_id>/availability-report', methods=['GET'])
def get_availability_report(event_id):
    """Get availability report for an event"""
    try:
        report = Event.get_availability_report(event_id)
        return jsonify({
            'success': True,
            'data': report
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@events_bp.route('/events/<int:event_id>/availability', methods=['POST'])
def update_availability():
    """Update user availability for proposed dates"""
    try:
        event_id = request.view_args['event_id']
        data = request.get_json()
        
        required_fields = ['user_id', 'proposed_date_id', 'is_available']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'{field} is required'
                }), 400
        
        success = Event.update_user_availability(
            user_id=data['user_id'],
            event_id=event_id,
            proposed_date_id=data['proposed_date_id'],
            is_available=data['is_available'],
            notes=data.get('notes')
        )
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Availability updated successfully'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to update availability'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@events_bp.route('/events/<int:event_id>/users/<int:user_id>/availability', methods=['GET'])
def get_user_availability(event_id, user_id):
    """Get user's availability for an event"""
    try:
        availability = Event.get_user_availability_for_event(user_id, event_id)
        return jsonify({
            'success': True,
            'data': availability
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500