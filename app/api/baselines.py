"""REST API endpoints for baseline vitals"""
from flask import Blueprint, request, jsonify
from datetime import datetime
from app.models.anomaly_log import db, PatientBaseline

baselines_bp = Blueprint('baselines', __name__, url_prefix='/api/baselines')


@baselines_bp.route('', methods=['GET'])
def get_all_baselines():
    """
    Fetch baseline vitals for all patients
    """
    try:
        baselines = PatientBaseline.query.all()
        
        return jsonify({
            'success': True,
            'count': len(baselines),
            'data': [baseline.to_dict() for baseline in baselines]
        }), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500


@baselines_bp.route('/<patient_id>', methods=['GET'])
def get_patient_baseline(patient_id):
    """
    Fetch baseline vitals for a specific patient
    """
    try:
        baseline = PatientBaseline.query.filter_by(patient_id=patient_id).first()
        
        if not baseline:
            return jsonify({
                'success': False,
                'message': f'No baseline found for patient {patient_id}'
            }), 404
        
        return jsonify({
            'success': True,
            'data': baseline.to_dict()
        }), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500


@baselines_bp.route('/<patient_id>', methods=['POST', 'PUT'])
def update_patient_baseline(patient_id):
    """
    Update or create baseline vitals for a patient
    
    JSON body:
    {
        "hr_mean": 75.0,
        "hr_std": 5.0,
        "spo2_mean": 97.0,
        "spo2_std": 1.5,
        "temperature_mean": 36.8,
        "temperature_std": 0.3,
        "sysbp_mean": 120.0,
        "sysbp_std": 8.0,
        "diabp_mean": 80.0,
        "diabp_std": 5.0
    }
    """
    try:
        data = request.get_json()
        
        baseline = PatientBaseline.query.filter_by(patient_id=patient_id).first()
        
        if not baseline:
            baseline = PatientBaseline(patient_id=patient_id)
        
        # Update fields
        if 'hr_mean' in data:
            baseline.hr_mean = data['hr_mean']
        if 'hr_std' in data:
            baseline.hr_std = data['hr_std']
        
        if 'spo2_mean' in data:
            baseline.spo2_mean = data['spo2_mean']
        if 'spo2_std' in data:
            baseline.spo2_std = data['spo2_std']
        
        if 'temperature_mean' in data:
            baseline.temperature_mean = data['temperature_mean']
        if 'temperature_std' in data:
            baseline.temperature_std = data['temperature_std']
        
        if 'sysbp_mean' in data:
            baseline.sysbp_mean = data['sysbp_mean']
        if 'sysbp_std' in data:
            baseline.sysbp_std = data['sysbp_std']
        
        if 'diabp_mean' in data:
            baseline.diabp_mean = data['diabp_mean']
        if 'diabp_std' in data:
            baseline.diabp_std = data['diabp_std']
        
        if 'samples_count' in data:
            baseline.samples_count = data['samples_count']
        
        baseline.last_updated = datetime.utcnow()
        
        db.session.add(baseline)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Baseline updated successfully',
            'data': baseline.to_dict()
        }), 200
    
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500


@baselines_bp.route('/<patient_id>', methods=['DELETE'])
def delete_patient_baseline(patient_id):
    """
    Delete baseline for a patient
    """
    try:
        baseline = PatientBaseline.query.filter_by(patient_id=patient_id).first()
        
        if not baseline:
            return jsonify({
                'success': False,
                'message': f'No baseline found for patient {patient_id}'
            }), 404
        
        db.session.delete(baseline)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Baseline deleted for patient {patient_id}'
        }), 200
    
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500


@baselines_bp.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        'success': True,
        'status': 'healthy',
        'message': 'Baselines API is running'
    }), 200
