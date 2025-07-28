from flask import Blueprint, request, jsonify
from models.user import User

users_bp = Blueprint('users', __name__)

@users_bp.route('/users', methods=['GET'])
def get_all_users():
    """Get all users"""
    try:
        users = User.get_all()
        return jsonify({
            'success': True,
            'data': users
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@users_bp.route('/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    """Get user by ID"""
    try:
        user = User.get_by_id(user_id)
        if user:
            return jsonify({
                'success': True,
                'data': user
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'User not found'
            }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@users_bp.route('/users', methods=['POST'])
def create_user():
    """Create a new user"""
    try:
        data = request.get_json()
        
        if not data or not data.get('name') or not data.get('email'):
            return jsonify({
                'success': False,
                'error': 'Name and email are required'
            }), 400
        
        user_id = User.create(
            name=data['name'],
            email=data['email'],
            phone=data.get('phone')
        )
        
        if user_id:
            user = User.get_by_id(user_id)
            return jsonify({
                'success': True,
                'data': user
            }), 201
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to create user'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@users_bp.route('/users/<int:user_id>', methods=['PUT'])
def update_user(user_id):
    """Update user information"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        success = User.update(
            user_id=user_id,
            name=data.get('name'),
            email=data.get('email'),
            phone=data.get('phone')
        )
        
        if success:
            user = User.get_by_id(user_id)
            return jsonify({
                'success': True,
                'data': user
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'User not found or update failed'
            }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@users_bp.route('/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    """Delete a user"""
    try:
        success = User.delete(user_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'User deleted successfully'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'User not found'
            }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500