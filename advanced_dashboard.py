#!/usr/bin/env python
"""Advanced Healthcare Anomaly Detection Dashboard"""

import os
os.environ['FLASK_ENV'] = 'development'

from flask import Flask, render_template_string, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta
import random

app = Flask(__name__)
CORS(app)

# Generate sample vital signs data
def generate_sample_data():
    patients = ['PAT_001', 'PAT_002', 'PAT_003', 'PAT_004', 'PAT_005']
    data = []
    base_time = datetime.now() - timedelta(hours=24)
    
    for i in range(100):
        timestamp = base_time + timedelta(minutes=i*15)
        patient = random.choice(patients)
        hr = random.randint(55, 120)
        spo2 = random.randint(92, 100)
        temp = round(random.uniform(36.5, 38.5), 1)
        sys_bp = random.randint(100, 160)
        dia_bp = random.randint(60, 100)
        score = round(random.uniform(0.1, 0.95), 2)
        
        if score > 0.7:
            severity = 'HIGH'
        elif score > 0.5:
            severity = 'MEDIUM'
        else:
            severity = 'LOW'
        
        data.append({
            'timestamp': timestamp.isoformat(),
            'patient_id': patient,
            'score': score,
            'severity': severity,
            'hr': hr,
            'spo2': spo2,
            'temp': temp,
            'sys_bp': sys_bp,
            'dia_bp': dia_bp
        })
    
    return data

sample_data = generate_sample_data()

HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Healthcare Anomaly Detection Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e0e0e0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            padding: 20px;
        }
        
        .container {
            max-width: 1600px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            margin-bottom: 40px;
            border-bottom: 2px solid #2d3561;
            padding-bottom: 20px;
        }
        
        .header h1 {
            font-size: 2.5em;
            color: #00d4ff;
            text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
            margin-bottom: 10px;
        }
        
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        
        .kpi-card {
            background: linear-gradient(135deg, #16213e 0%, #0f3460 100%);
            border: 2px solid #2d3561;
            border-radius: 12px;
            padding: 25px;
            text-align: center;
            transition: all 0.3s ease;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        .kpi-card:hover {
            transform: translateY(-5px);
            border-color: #00d4ff;
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
        }
        
        .kpi-value {
            font-size: 2.5em;
            font-weight: bold;
            color: #00d4ff;
            margin: 10px 0;
        }
        
        .kpi-label {
            font-size: 0.9em;
            color: #a0a0a0;
        }
        
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        
        .chart-card {
            background: linear-gradient(135deg, #16213e 0%, #0f3460 100%);
            border: 2px solid #2d3561;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        .chart-card h3 {
            color: #00d4ff;
            margin-bottom: 20px;
            text-align: center;
        }
        
        .chart-container {
            position: relative;
            height: 300px;
        }
        
        .data-table {
            width: 100%;
            background: linear-gradient(135deg, #16213e 0%, #0f3460 100%);
            border: 2px solid #2d3561;
            border-radius: 12px;
            overflow: hidden;
            margin-top: 40px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        .data-table h3 {
            color: #00d4ff;
            padding: 20px;
            border-bottom: 2px solid #2d3561;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
        }
        
        th {
            background: #0f3460;
            color: #00d4ff;
            padding: 15px;
            text-align: left;
            border-bottom: 2px solid #2d3561;
            font-weight: 600;
        }
        
        td {
            padding: 12px 15px;
            border-bottom: 1px solid #2d3561;
        }
        
        tr:hover {
            background: #1a3a52;
        }
        
        .badge {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: bold;
        }
        
        .badge-high {
            background: rgba(255, 107, 107, 0.2);
            color: #ff6b6b;
            border: 1px solid #ff6b6b;
        }
        
        .badge-medium {
            background: rgba(255, 217, 61, 0.2);
            color: #ffd93d;
            border: 1px solid #ffd93d;
        }
        
        .badge-low {
            background: rgba(149, 225, 211, 0.2);
            color: #95e1d3;
            border: 1px solid #95e1d3;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Hospital Anomaly Detection Dashboard</h1>
            <p>Real-time monitoring of patient vital signs and anomalies</p>
        </div>
        
        <!-- KPI Cards -->
        <div class="kpi-grid">
            <div class="kpi-card">
                <div>TOTAL ANOMALIES</div>
                <div class="kpi-value" id="total-anomalies">0</div>
                <div class="kpi-label">24 Hour Detection</div>
            </div>
            <div class="kpi-card">
                <div>HIGH ALERTS</div>
                <div class="kpi-value" id="high-alerts">0</div>
                <div class="kpi-label">Critical Events</div>
            </div>
            <div class="kpi-card">
                <div>AVG SCORE</div>
                <div class="kpi-value" id="avg-score">0.0</div>
                <div class="kpi-label">Anomaly Level</div>
            </div>
            <div class="kpi-card">
                <div>LAST ALERT</div>
                <div class="kpi-value" id="last-alert">--:--</div>
                <div class="kpi-label">Time (PM)</div>
            </div>
        </div>
        
        <!-- Charts -->
        <div class="charts-grid">
            <div class="chart-card">
                <h3>Severity Distribution</h3>
                <div class="chart-container">
                    <canvas id="severityChart"></canvas>
                </div>
            </div>
            
            <div class="chart-card">
                <h3>Anomaly Score Trend</h3>
                <div class="chart-container">
                    <canvas id="trendChart"></canvas>
                </div>
            </div>
            
            <div class="chart-card">
                <h3>Heart Rate Monitoring</h3>
                <div class="chart-container">
                    <canvas id="hrChart"></canvas>
                </div>
            </div>
            
            <div class="chart-card">
                <h3>SpO2 Level</h3>
                <div class="chart-container">
                    <canvas id="spo2Chart"></canvas>
                </div>
            </div>
            
            <div class="chart-card">
                <h3>Temperature Behavior</h3>
                <div class="chart-container">
                    <canvas id="tempChart"></canvas>
                </div>
            </div>
            
            <div class="chart-card">
                <h3>Blood Pressure Range</h3>
                <div class="chart-container">
                    <canvas id="bpChart"></canvas>
                </div>
            </div>
        </div>
        
        <!-- Data Table -->
        <div class="data-table">
            <h3>Recent Vital Signs Readings</h3>
            <table>
                <thead>
                    <tr>
                        <th>TIMESTAMP</th>
                        <th>PATIENT ID</th>
                        <th>SCORE</th>
                        <th>SEVERITY</th>
                        <th>HEART RATE</th>
                        <th>SpO2</th>
                        <th>TEMPERATURE</th>
                        <th>BLOOD PRESSURE</th>
                    </tr>
                </thead>
                <tbody id="dataTable"></tbody>
            </table>
        </div>
    </div>
    
    <script>
        const chartColors = {
            high: 'rgba(255, 107, 107, 0.8)',
            medium: 'rgba(255, 217, 61, 0.8)',
            low: 'rgba(149, 225, 211, 0.8)',
            grid: 'rgba(45, 53, 97, 0.3)',
            text: '#e0e0e0'
        };
        
        Chart.defaults.color = chartColors.text;
        Chart.defaults.borderColor = chartColors.grid;
        
        let charts = {};
        
        async function loadDashboard() {
            const response = await fetch('/api/dashboard-data');
            const data = await response.json();
            
            // Update KPIs
            document.getElementById('total-anomalies').textContent = data.total_anomalies;
            document.getElementById('high-alerts').textContent = data.high_alerts;
            document.getElementById('avg-score').textContent = data.avg_score.toFixed(2);
            document.getElementById('last-alert').textContent = data.last_alert;
            
            // Severity Distribution Chart
            const severityCtx = document.getElementById('severityChart').getContext('2d');
            if (charts.severity) charts.severity.destroy();
            charts.severity = new Chart(severityCtx, {
                type: 'doughnut',
                data: {
                    labels: ['HIGH', 'MEDIUM', 'LOW'],
                    datasets: [{
                        data: [data.high_count, data.medium_count, data.low_count],
                        backgroundColor: [chartColors.high, chartColors.medium, chartColors.low],
                        borderColor: '#0f3460',
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'bottom' }
                    }
                }
            });
            
            // Anomaly Score Trend
            const trendCtx = document.getElementById('trendChart').getContext('2d');
            if (charts.trend) charts.trend.destroy();
            charts.trend = new Chart(trendCtx, {
                type: 'line',
                data: {
                    labels: data.timestamps.slice(-24),
                    datasets: [{
                        label: 'Anomaly Score',
                        data: data.scores.slice(-24),
                        borderColor: '#00d4ff',
                        backgroundColor: 'rgba(0, 212, 255, 0.1)',
                        tension: 0.4,
                        fill: true,
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: true } },
                    scales: {
                        y: { min: 0, max: 1, ticks: { color: chartColors.text } }
                    }
                }
            });
            
            // Heart Rate Chart
            const hrCtx = document.getElementById('hrChart').getContext('2d');
            if (charts.hr) charts.hr.destroy();
            charts.hr = new Chart(hrCtx, {
                type: 'line',
                data: {
                    labels: data.timestamps.slice(-24),
                    datasets: [{
                        label: 'Heart Rate (bpm)',
                        data: data.hr_values.slice(-24),
                        borderColor: '#ff6b6b',
                        backgroundColor: 'rgba(255, 107, 107, 0.1)',
                        tension: 0.4,
                        fill: true,
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: { y: { ticks: { color: chartColors.text } } }
                }
            });
            
            // SpO2 Chart
            const spo2Ctx = document.getElementById('spo2Chart').getContext('2d');
            if (charts.spo2) charts.spo2.destroy();
            charts.spo2 = new Chart(spo2Ctx, {
                type: 'line',
                data: {
                    labels: data.timestamps.slice(-24),
                    datasets: [{
                        label: 'SpO2 (%)',
                        data: data.spo2_values.slice(-24),
                        borderColor: '#95e1d3',
                        backgroundColor: 'rgba(149, 225, 211, 0.1)',
                        tension: 0.4,
                        fill: true,
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: { y: { min: 85, max: 100, ticks: { color: chartColors.text } } }
                }
            });
            
            // Temperature Chart
            const tempCtx = document.getElementById('tempChart').getContext('2d');
            if (charts.temp) charts.temp.destroy();
            charts.temp = new Chart(tempCtx, {
                type: 'bar',
                data: {
                    labels: data.timestamps.slice(-12),
                    datasets: [{
                        label: 'Temperature (C)',
                        data: data.temp_values.slice(-12),
                        backgroundColor: 'rgba(255, 217, 61, 0.7)',
                        borderColor: '#ffd93d',
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: { y: { ticks: { color: chartColors.text } } }
                }
            });
            
            // Blood Pressure Chart
            const bpCtx = document.getElementById('bpChart').getContext('2d');
            if (charts.bp) charts.bp.destroy();
            charts.bp = new Chart(bpCtx, {
                type: 'line',
                data: {
                    labels: data.timestamps.slice(-24),
                    datasets: [
                        {
                            label: 'Systolic',
                            data: data.sys_bp_values.slice(-24),
                            borderColor: '#ff6b6b',
                            tension: 0.4,
                            borderWidth: 2
                        },
                        {
                            label: 'Diastolic',
                            data: data.dia_bp_values.slice(-24),
                            borderColor: '#95e1d3',
                            tension: 0.4,
                            borderWidth: 2
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: { y: { ticks: { color: chartColors.text } } }
                }
            });
            
            // Populate data table
            const tbody = document.getElementById('dataTable');
            tbody.innerHTML = '';
            data.table_data.forEach(row => {
                const tr = document.createElement('tr');
                const badgeClass = 'badge-' + row.severity.toLowerCase();
                
                tr.innerHTML = '<td>' + row.timestamp.split('T')[1] + '</td>' +
                    '<td>' + row.patient_id + '</td>' +
                    '<td>' + row.score.toFixed(2) + '</td>' +
                    '<td><span class="badge ' + badgeClass + '">' + row.severity + '</span></td>' +
                    '<td>' + row.hr + ' bpm</td>' +
                    '<td>' + row.spo2 + '%</td>' +
                    '<td>' + row.temp + 'C</td>' +
                    '<td>' + row.sys_bp + '/' + row.dia_bp + '</td>';
                tbody.appendChild(tr);
            });
            
            // Auto-refresh every 10 seconds
            setTimeout(loadDashboard, 10000);
        }
        
        loadDashboard();
    </script>
</body>
</html>'''

@app.route('/')
def dashboard():
    return render_template_string(HTML)

@app.route('/api/dashboard-data')
def get_dashboard_data():
    high_count = len([d for d in sample_data if d['severity'] == 'HIGH'])
    medium_count = len([d for d in sample_data if d['severity'] == 'MEDIUM'])
    low_count = len([d for d in sample_data if d['severity'] == 'LOW'])
    avg_score = sum(d['score'] for d in sample_data) / len(sample_data)
    
    return jsonify({
        'total_anomalies': len(sample_data),
        'high_alerts': high_count,
        'high_count': high_count,
        'medium_count': medium_count,
        'low_count': low_count,
        'avg_score': avg_score,
        'last_alert': sample_data[-1]['timestamp'].split('T')[1][:5],
        'timestamps': [d['timestamp'].split('T')[1][:5] for d in sample_data],
        'scores': [d['score'] for d in sample_data],
        'hr_values': [d['hr'] for d in sample_data],
        'spo2_values': [d['spo2'] for d in sample_data],
        'temp_values': [d['temp'] for d in sample_data],
        'sys_bp_values': [d['sys_bp'] for d in sample_data],
        'dia_bp_values': [d['dia_bp'] for d in sample_data],
        'table_data': [
            {
                'timestamp': d['timestamp'],
                'patient_id': d['patient_id'],
                'score': d['score'],
                'severity': d['severity'],
                'hr': d['hr'],
                'spo2': d['spo2'],
                'temp': d['temp'],
                'sys_bp': d['sys_bp'],
                'dia_bp': d['dia_bp']
            }
            for d in sample_data[-20:]
        ]
    })

if __name__ == '__main__':
    print('')
    print('='*70)
    print('ADVANCED HEALTHCARE ANOMALY DETECTION DASHBOARD')
    print('='*70)
    print('')
    print('OPEN YOUR BROWSER:')
    print('  http://localhost:5000')
    print('')
    print('FEATURES:')
    print('  [+] Real-time KPI cards')
    print('  [+] Interactive charts (Severity, Trend, HR, SpO2, Temp, BP)')
    print('  [+] Live data table with vital signs')
    print('  [+] Auto-refresh every 10 seconds')
    print('  [+] Professional dark theme UI')
    print('')
    print('Press Ctrl+C to stop')
    print('='*70)
    print('')
    
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
