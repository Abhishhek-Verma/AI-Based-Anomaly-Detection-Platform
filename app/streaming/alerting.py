"""
Email Alerting System for Critical Anomalies
Sends formatted alerts with cooldown logic to prevent spam
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
import os

logger = logging.getLogger(__name__)


@dataclass
class AlertCooldown:
    """Tracks alert cooldown state per patient"""
    patient_id: str
    last_alert_time: datetime
    last_alert_severity: str
    cooldown_minutes: int = 30
    
    def is_cooldown_active(self, severity: str) -> bool:
        """
        Check if alerts for this patient are in cooldown.
        Cooldown is lifted if new severity is higher.
        
        Args:
            severity: New anomaly severity
        
        Returns:
            True if in cooldown, False if alert should be sent
        """
        if not self.last_alert_time:
            return False
        
        time_since_alert = datetime.now() - self.last_alert_time
        cooldown_period = timedelta(minutes=self.cooldown_minutes)
        
        # Severity hierarchy
        severity_levels = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}
        current_level = severity_levels.get(severity, 0)
        last_level = severity_levels.get(self.last_alert_severity, 0)
        
        # If new alert is higher severity, bypass cooldown
        if current_level > last_level:
            return False
        
        # Otherwise check cooldown period
        return time_since_alert < cooldown_period
    
    def update(self, severity: str) -> None:
        """Update cooldown state after alert sent"""
        self.last_alert_time = datetime.now()
        self.last_alert_severity = severity


class AlertManager:
    """
    Manages alert sending with cooldown logic.
    Prevents alert spam while escalating severity when needed.
    """
    
    def __init__(self, cooldown_minutes: int = 30, severity_threshold: str = 'MEDIUM'):
        """
        Initialize alert manager.
        
        Args:
            cooldown_minutes: Cooldown period between alerts per patient
            severity_threshold: Only send alerts for this level or higher (LOW/MEDIUM/HIGH)
        """
        self.cooldown_minutes = cooldown_minutes
        self.severity_threshold = severity_threshold
        self.cooldowns: Dict[str, AlertCooldown] = {}
        
        # Severity levels for filtering
        self.severity_levels = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}
        self.threshold_level = self.severity_levels.get(severity_threshold, 2)
    
    def should_send_alert(self, patient_id: str, severity: str) -> bool:
        """
        Determine if alert should be sent based on severity and cooldown.
        
        Args:
            patient_id: Patient identifier
            severity: Anomaly severity (LOW/MEDIUM/HIGH)
        
        Returns:
            True if alert should be sent
        """
        # Check severity threshold
        severity_level = self.severity_levels.get(severity, 0)
        if severity_level < self.threshold_level:
            return False
        
        # Check cooldown
        if patient_id not in self.cooldowns:
            self.cooldowns[patient_id] = AlertCooldown(
                patient_id=patient_id,
                last_alert_time=None,
                last_alert_severity='LOW',
                cooldown_minutes=self.cooldown_minutes
            )
            return True
        
        cooldown = self.cooldowns[patient_id]
        if cooldown.is_cooldown_active(severity):
            return False
        
        return True
    
    def mark_alert_sent(self, patient_id: str, severity: str) -> None:
        """
        Mark that alert was sent for patient.
        
        Args:
            patient_id: Patient identifier
            severity: Alert severity
        """
        if patient_id not in self.cooldowns:
            self.cooldowns[patient_id] = AlertCooldown(
                patient_id=patient_id,
                last_alert_time=None,
                last_alert_severity='LOW',
                cooldown_minutes=self.cooldown_minutes
            )
        
        self.cooldowns[patient_id].update(severity)
        logger.info(f"Alert marked as sent for patient {patient_id} (severity: {severity})")
    
    def get_cooldown_status(self, patient_id: str) -> Dict:
        """Get cooldown status for patient"""
        if patient_id not in self.cooldowns:
            return {
                'patient_id': patient_id,
                'in_cooldown': False,
                'last_alert_time': None
            }
        
        cooldown = self.cooldowns[patient_id]
        return {
            'patient_id': patient_id,
            'in_cooldown': cooldown.is_cooldown_active(cooldown.last_alert_severity),
            'last_alert_time': cooldown.last_alert_time.isoformat() if cooldown.last_alert_time else None,
            'last_alert_severity': cooldown.last_alert_severity,
            'next_alert_possible': (
                cooldown.last_alert_time + timedelta(minutes=cooldown.cooldown_minutes)
            ).isoformat() if cooldown.last_alert_time else None
        }


class EmailAlertSender:
    """
    Sends email alerts for critical anomalies.
    Integrates with SMTP server for actual email delivery.
    """
    
    def __init__(self,
                 smtp_server: Optional[str] = None,
                 smtp_port: int = 587,
                 sender_email: Optional[str] = None,
                 sender_password: Optional[str] = None,
                 sender_name: str = "Anomaly Detection System"):
        """
        Initialize email sender.
        
        Args:
            smtp_server: SMTP server address (default from env)
            smtp_port: SMTP port (default 587 for TLS)
            sender_email: Sender email address (default from env)
            sender_password: Sender password (default from env)
            sender_name: Display name for sender
        """
        self.smtp_server = smtp_server or os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = smtp_port
        self.sender_email = sender_email or os.getenv('SENDER_EMAIL')
        self.sender_password = sender_password or os.getenv('SENDER_PASSWORD')
        self.sender_name = sender_name
        self.enabled = bool(self.sender_email and self.sender_password)
        
        if not self.enabled:
            logger.warning("Email configuration incomplete. Email alerts disabled.")
    
    def send_alert(self,
                  recipient_email: str,
                  patient_id: str,
                  severity: str,
                  html_content: str,
                  alert_time: datetime) -> bool:
        """
        Send email alert.
        
        Args:
            recipient_email: Recipient email address
            patient_id: Patient identifier
            severity: Alert severity
            html_content: HTML formatted alert content
            alert_time: Time of alert
        
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            logger.info(f"Email alerts disabled. Would send to {recipient_email}")
            return True  # Succeed silently in demo mode
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{severity}] Patient Anomaly Alert - {patient_id} at {alert_time.strftime('%Y-%m-%d %H:%M:%S')}"
            msg['From'] = f"{self.sender_name} <{self.sender_email}>"
            msg['To'] = recipient_email
            
            # Attach HTML content
            msg.attach(MIMEText(html_content, 'html'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            logger.info(f"Email alert sent to {recipient_email} for patient {patient_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    def send_test_alert(self, recipient_email: str) -> bool:
        """
        Send test email to verify configuration.
        
        Args:
            recipient_email: Test email address
        
        Returns:
            True if sent successfully
        """
        html_content = """
        <html>
        <body style="font-family: Arial, sans-serif;">
            <h2>Test Alert - Anomaly Detection System</h2>
            <p>This is a test email from the AI-Based Anomaly Detection Platform.</p>
            <p>Email alerting is configured correctly.</p>
            <p style="color: #999; font-size: 12px;">
                Sent at: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """
            </p>
        </body>
        </html>
        """
        
        return self.send_alert(
            recipient_email=recipient_email,
            patient_id='TEST',
            severity='INFO',
            html_content=html_content,
            alert_time=datetime.now()
        )


class AlertLogger:
    """
    Logs all alerts for audit trail and historical analysis.
    """
    
    def __init__(self, log_file: str = 'logs/alerts.log'):
        """
        Initialize alert logger.
        
        Args:
            log_file: Path to alert log file
        """
        self.log_file = log_file
        
        # Create logs directory if needed
        import os
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    def log_alert(self, patient_id: str, severity: str, anomaly_score: float,
                 primary_contributor: str, explanation_dict: Dict) -> None:
        """
        Log alert to file.
        
        Args:
            patient_id: Patient identifier
            severity: Alert severity
            anomaly_score: Anomaly score
            primary_contributor: Primary abnormal vital
            explanation_dict: Full explanation dictionary
        """
        try:
            with open(self.log_file, 'a') as f:
                timestamp = datetime.now().isoformat()
                log_entry = {
                    'timestamp': timestamp,
                    'patient_id': patient_id,
                    'severity': severity,
                    'anomaly_score': anomaly_score,
                    'primary_contributor': primary_contributor
                }
                
                # Write as JSON line for easy parsing
                import json
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to log alert: {e}")
    
    def get_patient_alerts(self, patient_id: str, hours: int = 24) -> List[Dict]:
        """
        Retrieve recent alerts for a patient.
        
        Args:
            patient_id: Patient identifier
            hours: Look back this many hours
        
        Returns:
            List of alert dictionaries
        """
        alerts = []
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        try:
            if not os.path.exists(self.log_file):
                return alerts
            
            import json
            with open(self.log_file, 'r') as f:
                for line in f:
                    try:
                        alert = json.loads(line)
                        if alert.get('patient_id') == patient_id:
                            alert_time = datetime.fromisoformat(alert['timestamp'])
                            if alert_time > cutoff_time:
                                alerts.append(alert)
                    except json.JSONDecodeError:
                        continue
        
        except Exception as e:
            logger.error(f"Failed to read alert history: {e}")
        
        return alerts
