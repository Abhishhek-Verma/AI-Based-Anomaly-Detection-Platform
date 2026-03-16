"""REST API endpoints for anomaly logs"""
from flask import Blueprint, request, jsonify, current_app
from sqlalchemy import desc, and_
from datetime import datetime, timedelta
from app.models.anomaly_log import db, AnomalyLog, SeverityEnum

anomalies_bp = Blueprint('anomalies', __name__, url_prefix='/api/anomalies')


@anomalies_bp.route('', methods=['GET'])
def get_anomalies():
    """
    Fetch anomaly logs with optional filtering
    
    Query parameters:
    - patient_id: Filter by patient ID
    - severity: Filter by severity (HIGH, MEDIUM, LOW)
    - start_time: Start timestamp (ISO format)
    - end_time: End timestamp (ISO format)
    - page: Page number (default: 1)
    - per_page: Items per page (default: 50)
    """
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get(
            'per_page',
            current_app.config['ITEMS_PER_PAGE'],
            type=int
        )
        patient_id = request.args.get('patient_id', type=str)
        severity = request.args.get('severity', type=str)
        start_time = request.args.get('start_time', type=str)
        end_time = request.args.get('end_time', type=str)
        only_alerts = request.args.get('only_alerts', 'false').lower() == 'true'
        
        # Build query
        query = AnomalyLog.query
        
        if patient_id:
            query = query.filter_by(patient_id=patient_id)
        
        if severity:
            query = query.filter_by(severity=severity)
        
        if only_alerts:
            query = query.filter_by(alert_sent=True)
        
        if start_time:
            try:
                start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                query = query.filter(AnomalyLog.timestamp >= start_dt)
            except:
                pass
        
        if end_time:
            try:
                end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                query = query.filter(AnomalyLog.timestamp <= end_dt)
            except:
                pass
        
        # Sort by timestamp descending
        query = query.order_by(desc(AnomalyLog.timestamp))
        
        # Paginate
        pagination = query.paginate(page=page, per_page=per_page, error_out=False)
        
        anomalies = [anomaly.to_dict() for anomaly in pagination.items]
        
        return jsonify({
            'success': True,
            'data': anomalies,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': pagination.total,
                'pages': pagination.pages,
                'has_next': pagination.has_next,
                'has_prev': pagination.has_prev,
            }
        }), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error fetching anomalies: {str(e)}'
        }), 500


@anomalies_bp.route('/patient/<patient_id>', methods=['GET'])
def get_patient_anomalies(patient_id):
    """
    Fetch anomalies for a specific patient
    
    Query parameters:
    - limit: Number of results (default: 100)
    - severity: Filter by severity
    """
    try:
        limit = request.args.get('limit', 100, type=int)
        severity = request.args.get('severity', type=str)
        
        query = AnomalyLog.query.filter_by(patient_id=patient_id)
        
        if severity:
            query = query.filter_by(severity=severity)
        
        query = query.order_by(desc(AnomalyLog.timestamp)).limit(limit)
        anomalies = [anomaly.to_dict() for anomaly in query.all()]
        
        return jsonify({
            'success': True,
            'patient_id': patient_id,
            'count': len(anomalies),
            'data': anomalies
        }), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500


@anomalies_bp.route('/latest', methods=['GET'])
def get_latest_anomalies():
    """
    Fetch the latest anomalies (last 24 hours by default)
    
    Query parameters:
    - hours: Lookback hours (default: 24)
    - limit: Number of results (default: 50)
    """
    try:
        hours = request.args.get('hours', 24, type=int)
        limit = request.args.get('limit', 50, type=int)
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        anomalies = AnomalyLog.query.filter(
            AnomalyLog.timestamp >= cutoff_time
        ).order_by(
            desc(AnomalyLog.timestamp)
        ).limit(limit).all()
        
        return jsonify({
            'success': True,
            'lookback_hours': hours,
            'count': len(anomalies),
            'data': [anomaly.to_dict() for anomaly in anomalies]
        }), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500


@anomalies_bp.route('/summary', methods=['GET'])
def get_anomalies_summary():
    """
    Get summary statistics of anomalies
    
    Query parameters:
    - hours: Lookback hours (default: 24)
    """
    try:
        hours = request.args.get('hours', 24, type=int)
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        total_count = AnomalyLog.query.filter(
            AnomalyLog.timestamp >= cutoff_time
        ).count()
        
        high_count = AnomalyLog.query.filter(
            and_(
                AnomalyLog.timestamp >= cutoff_time,
                AnomalyLog.severity == SeverityEnum.HIGH
            )
        ).count()
        
        medium_count = AnomalyLog.query.filter(
            and_(
                AnomalyLog.timestamp >= cutoff_time,
                AnomalyLog.severity == SeverityEnum.MEDIUM
            )
        ).count()
        
        low_count = AnomalyLog.query.filter(
            and_(
                AnomalyLog.timestamp >= cutoff_time,
                AnomalyLog.severity == SeverityEnum.LOW
            )
        ).count()
        
        alert_count = AnomalyLog.query.filter(
            and_(
                AnomalyLog.timestamp >= cutoff_time,
                AnomalyLog.alert_sent == True
            )
        ).count()
        
        avg_score = db.session.query(
            db.func.avg(AnomalyLog.anomaly_score)
        ).filter(
            AnomalyLog.timestamp >= cutoff_time
        ).scalar() or 0
        
        return jsonify({
            'success': True,
            'lookback_hours': hours,
            'summary': {
                'total_anomalies': total_count,
                'high_severity': high_count,
                'medium_severity': medium_count,
                'low_severity': low_count,
                'alerts_sent': alert_count,
                'average_score': round(float(avg_score), 4),
            }
        }), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500


@anomalies_bp.route('/by-severity', methods=['GET'])
def get_anomalies_by_severity():
    """
    Get anomalies grouped by severity
    
    Query parameters:
    - hours: Lookback hours (default: 24)
    """
    try:
        hours = request.args.get('hours', 24, type=int)
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        severity_data = {}
        for severity in SeverityEnum:
            count = AnomalyLog.query.filter(
                and_(
                    AnomalyLog.timestamp >= cutoff_time,
                    AnomalyLog.severity == severity
                )
            ).count()
            severity_data[severity.value] = count
        
        return jsonify({
            'success': True,
            'lookback_hours': hours,
            'data': severity_data
        }), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500


@anomalies_bp.route('/store', methods=['POST'])
def store_anomaly():
    """
    Store a new anomaly in the database
    
    JSON body:
    {
        "patient_id": "PAT_0001",
        "anomaly_score": 0.85,
        "autoencoder_score": 0.82,
        "isolation_forest_score": 0.88,
        "severity": "HIGH",
        "vital_signs": {...},
        "baseline_vitals": {...},
        ...
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'patient_id' not in data:
            return jsonify({
                'success': False,
                'message': 'Missing required field: patient_id'
            }), 400
        
        anomaly = AnomalyLog(
            patient_id=data['patient_id'],
            anomaly_score=data.get('anomaly_score', 0),
            autoencoder_score=data.get('autoencoder_score', 0),
            isolation_forest_score=data.get('isolation_forest_score', 0),
            severity=data.get('severity', 'LOW'),
            vital_signs=data.get('vital_signs', {}),
            baseline_vitals=data.get('baseline_vitals', {}),
            vital_deviations=data.get('vital_deviations', {}),
            abnormal_vitals=data.get('abnormal_vitals', []),
            primary_contributor=data.get('primary_contributor'),
            primary_contributor_percentage=data.get('primary_contributor_percentage'),
            explanation=data.get('explanation'),
            recommendation=data.get('recommendation'),
            alert_sent=data.get('alert_sent', False),
            timestamp=datetime.utcnow(),
        )
        
        db.session.add(anomaly)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Anomaly stored successfully',
            'id': anomaly.id
        }), 201
    
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'message': f'Error storing anomaly: {str(e)}'
        }), 500
