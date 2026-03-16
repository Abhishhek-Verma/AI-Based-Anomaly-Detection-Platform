"""REST API endpoints for patient data"""
from flask import Blueprint, request, jsonify, current_app
from sqlalchemy import desc, func, and_
from datetime import datetime, timedelta
from app.models.anomaly_log import db, AnomalyLog, PatientBaseline, PatientStats, SeverityEnum

patients_bp = Blueprint('patients', __name__, url_prefix='/api/patients')


@patients_bp.route('', methods=['GET'])
def get_all_patients():
    """
    Fetch all patients with their latest statistics
    
    Query parameters:
    - sort_by: Sort field (anomaly_count, anomaly_rate, last_seen)
    - order: asc or desc (default: desc)
    """
    try:
        sort_by = request.args.get('sort_by', 'last_seen')
        order = request.args.get('order', 'desc').lower()
        
        # Get unique patients from anomaly logs
        patients_query = db.session.query(
            AnomalyLog.patient_id,
            func.count(AnomalyLog.id).label('anomaly_count'),
            func.count(
                AnomalyLog.id
            ).filter(AnomalyLog.severity == SeverityEnum.HIGH).label('high_severity_count'),
            func.max(AnomalyLog.timestamp).label('last_seen'),
            func.avg(AnomalyLog.anomaly_score).label('avg_score'),
        ).group_by(AnomalyLog.patient_id)
        
        # Apply sorting
        if sort_by == 'anomaly_count':
            if order == 'asc':
                patients_query = patients_query.order_by(func.count(AnomalyLog.id))
            else:
                patients_query = patients_query.order_by(desc(func.count(AnomalyLog.id)))
        elif sort_by == 'anomaly_rate':
            if order == 'asc':
                patients_query = patients_query.order_by(func.avg(AnomalyLog.anomaly_score))
            else:
                patients_query = patients_query.order_by(desc(func.avg(AnomalyLog.anomaly_score)))
        else:  # last_seen
            if order == 'asc':
                patients_query = patients_query.order_by(func.max(AnomalyLog.timestamp))
            else:
                patients_query = patients_query.order_by(desc(func.max(AnomalyLog.timestamp)))
        
        patients = []
        for row in patients_query.all():
            patient_data = {
                'patient_id': row.patient_id,
                'total_anomalies': row.anomaly_count or 0,
                'high_severity_count': row.high_severity_count or 0,
                'last_seen': row.last_seen.isoformat() if row.last_seen else None,
                'average_anomaly_score': round(float(row.avg_score), 4) if row.avg_score else 0,
            }
            patients.append(patient_data)
        
        return jsonify({
            'success': True,
            'count': len(patients),
            'data': patients
        }), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error fetching patients: {str(e)}'
        }), 500


@patients_bp.route('/<patient_id>', methods=['GET'])
def get_patient_details(patient_id):
    """
    Fetch detailed information about a specific patient
    """
    try:
        # Get patient statistics
        patient_stats = db.session.query(
            func.count(AnomalyLog.id).label('total_anomalies'),
            func.count(
                AnomalyLog.id
            ).filter(AnomalyLog.severity == SeverityEnum.HIGH).label('high_severity'),
            func.count(
                AnomalyLog.id
            ).filter(AnomalyLog.severity == SeverityEnum.MEDIUM).label('medium_severity'),
            func.count(
                AnomalyLog.id
            ).filter(AnomalyLog.severity == SeverityEnum.LOW).label('low_severity'),
            func.max(AnomalyLog.timestamp).label('last_anomaly'),
            func.avg(AnomalyLog.anomaly_score).label('avg_score'),
        ).filter(AnomalyLog.patient_id == patient_id).first()
        
        # Get patient baseline
        baseline = PatientBaseline.query.filter_by(patient_id=patient_id).first()
        baseline_data = baseline.to_dict() if baseline else None
        
        # Get latest anomaly
        latest_anomaly = AnomalyLog.query.filter_by(
            patient_id=patient_id
        ).order_by(desc(AnomalyLog.timestamp)).first()
        latest_anomaly_data = latest_anomaly.to_dict() if latest_anomaly else None
        
        # Get last 10 vital readings
        recent_vitals = AnomalyLog.query.filter_by(
            patient_id=patient_id
        ).order_by(desc(AnomalyLog.timestamp)).limit(10).all()
        
        vital_history = []
        for vital in reversed(recent_vitals):
            vital_history.append({
                'timestamp': vital.timestamp.isoformat(),
                'vital_signs': vital.vital_signs,
                'severity': vital.severity.value,
                'anomaly_score': vital.anomaly_score,
            })
        
        return jsonify({
            'success': True,
            'patient_id': patient_id,
            'statistics': {
                'total_anomalies': patient_stats.total_anomalies or 0,
                'high_severity': patient_stats.high_severity or 0,
                'medium_severity': patient_stats.medium_severity or 0,
                'low_severity': patient_stats.low_severity or 0,
                'last_anomaly': patient_stats.last_anomaly.isoformat() if patient_stats.last_anomaly else None,
                'average_score': round(float(patient_stats.avg_score), 4) if patient_stats.avg_score else 0,
            },
            'baseline': baseline_data,
            'latest_anomaly': latest_anomaly_data,
            'vital_history': vital_history,
        }), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error fetching patient details: {str(e)}'
        }), 500


@patients_bp.route('/<patient_id>/history', methods=['GET'])
def get_patient_history(patient_id):
    """
    Fetch detailed anomaly history for a patient
    
    Query parameters:
    - limit: Number of records (default: 100)
    - severity: Filter by severity
    - hours: Lookback hours (default: 168 for 7 days)
    """
    try:
        limit = request.args.get('limit', 100, type=int)
        severity = request.args.get('severity', type=str)
        hours = request.args.get('hours', 168, type=int)
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        query = AnomalyLog.query.filter(
            and_(
                AnomalyLog.patient_id == patient_id,
                AnomalyLog.timestamp >= cutoff_time
            )
        )
        
        if severity:
            query = query.filter_by(severity=severity)
        
        anomalies = query.order_by(desc(AnomalyLog.timestamp)).limit(limit).all()
        
        return jsonify({
            'success': True,
            'patient_id': patient_id,
            'lookback_hours': hours,
            'count': len(anomalies),
            'data': [anomaly.to_dict() for anomaly in anomalies]
        }), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500


@patients_bp.route('/<patient_id>/vitals-trend', methods=['GET'])
def get_patient_vitals_trend(patient_id):
    """
    Fetch vital signs trend for a patient
    
    Query parameters:
    - vital: Specific vital (hr, spo2, temperature, sysbp, diabp)
    - limit: Number of points (default: 100)
    """
    try:
        vital = request.args.get('vital', type=str)
        limit = request.args.get('limit', 100, type=int)
        
        anomalies = AnomalyLog.query.filter_by(
            patient_id=patient_id
        ).order_by(AnomalyLog.timestamp).limit(limit).all()
        
        trend_data = []
        for anomaly in anomalies:
            vital_signs = anomaly.vital_signs or {}
            baseline_vitals = anomaly.baseline_vitals or {}
            
            # Map vital names
            vital_mapping = {
                'hr': ('HR', 'HR'),
                'spo2': ('SpO2', 'SpO2'),
                'temperature': ('Temperature', 'Temperature'),
                'sysbp': ('SysBP', 'SysBP'),
                'diabp': ('DiaBP', 'DiaBP'),
            }
            
            if vital and vital in vital_mapping:
                current_key, baseline_key = vital_mapping[vital]
                current_value = vital_signs.get(current_key)
                baseline_value = baseline_vitals.get(baseline_key)
                
                if current_value is not None:
                    trend_data.append({
                        'timestamp': anomaly.timestamp.isoformat(),
                        'current': current_value,
                        'baseline': baseline_value,
                        'severity': anomaly.severity.value,
                    })
            else:
                # Return all vitals
                trend_data.append({
                    'timestamp': anomaly.timestamp.isoformat(),
                    'vitals': vital_signs,
                    'baseline': baseline_vitals,
                    'severity': anomaly.severity.value,
                })
        
        return jsonify({
            'success': True,
            'patient_id': patient_id,
            'vital': vital,
            'data': trend_data
        }), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500
