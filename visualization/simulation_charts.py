"""
Charts and visualization module for simulation results
"""
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import pytz

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.patches import Patch

logger = logging.getLogger(__name__)


class SimulationChartsGenerator:
    """Generate charts and visualizations for simulation results."""
    
    def __init__(self, timezone: str = 'UTC'):
        self.timezone = timezone
        self.tz = pytz.timezone(timezone)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def create_all_charts(self, simulation_id: str, results_dir: Path):
        """Create all charts for a simulation."""
        # Load results
        results_file = results_dir / f'{simulation_id}_results.json'
        timeline_file = results_dir / f'{simulation_id}_timeline.csv'
        
        if not results_file.exists():
            logger.error(f"Results file not found: {results_file}")
            return
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        if timeline_file.exists():
            timeline_df = pd.read_csv(timeline_file)
            timeline_df['timestamp'] = pd.to_datetime(timeline_df['timestamp'])
        else:
            timeline_df = pd.DataFrame()
        
        # Create visualizations
        self.create_alert_timeline_chart(results, timeline_df, results_dir / f'{simulation_id}_alert_timeline.png')
        self.create_accuracy_metrics_chart(results, results_dir / f'{simulation_id}_accuracy_metrics.png')
        self.create_vehicle_performance_chart(results, results_dir / f'{simulation_id}_vehicle_performance.png')
        self.create_confusion_matrix_chart(results, results_dir / f'{simulation_id}_confusion_matrix.png')
        
        if not timeline_df.empty:
            self.create_trip_completion_chart(timeline_df, results_dir / f'{simulation_id}_trip_completion.png')
        
        logger.info(f"Charts created in {results_dir}")
    
    def create_alert_timeline_chart(self, results: Dict, timeline_df: pd.DataFrame, output_file: Path):
        """Create timeline chart showing when alerts were generated."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Parse alert timeline
        alerts = results.get('alert_timeline', [])
        if not alerts:
            ax.text(0.5, 0.5, 'No alerts generated during simulation', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Alert Timeline - No Alerts')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            return
        
        # Convert to DataFrame
        alert_df = pd.DataFrame(alerts)
        alert_df['timestamp'] = pd.to_datetime(alert_df['timestamp'])
        alert_df['timestamp_local'] = alert_df['timestamp'].dt.tz_convert(self.tz)
        
        # Group by hour and alert type
        alert_df['hour'] = alert_df['timestamp_local'].dt.floor('H')
        hourly_alerts = alert_df.groupby(['hour', 'alert_type']).size().unstack(fill_value=0)
        
        # Create stacked bar chart
        colors = {'low': '#2ecc71', 'medium': '#f39c12', 'high': '#e74c3c', 'critical': '#c0392b'}
        hourly_alerts.plot(kind='bar', stacked=True, ax=ax, 
                          color=[colors.get(col, '#95a5a6') for col in hourly_alerts.columns],
                          width=0.8)
        
        ax.set_title(f'Alert Timeline - {results["simulation_id"]}', fontsize=16)
        ax.set_xlabel(f'Time ({self.timezone})', fontsize=12)
        ax.set_ylabel('Number of Alerts', fontsize=12)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M', tz=self.tz))
        plt.xticks(rotation=45, ha='right')
        
        # Add legend
        ax.legend(title='Alert Type', loc='upper right')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_accuracy_metrics_chart(self, results: Dict, output_file: Path):
        """Create chart showing prediction accuracy metrics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Metrics bar chart
        metrics = results.get('metrics', {})
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        metric_values = [
            metrics.get('accuracy', 0),
            metrics.get('precision', 0),
            metrics.get('recall', 0),
            metrics.get('f1_score', 0)
        ]
        
        bars = ax1.bar(metric_names, metric_values, color=['#3498db', '#9b59b6', '#e74c3c', '#2ecc71'])
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1%}', ha='center', va='bottom')
        
        ax1.set_ylim(0, 1.1)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('Prediction Accuracy Metrics', fontsize=14)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Alert distribution pie chart
        total_alerts = metrics.get('alerts_generated', 0)
        if total_alerts > 0:
            confusion = metrics.get('confusion_matrix', {})
            tp = confusion.get('true_positives', 0)
            fp = confusion.get('false_positives', 0)
            
            sizes = [tp, fp]
            labels = ['Correct Alerts', 'False Alarms']
            colors = ['#2ecc71', '#e74c3c']
            
            ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax2.set_title(f'Alert Accuracy (n={total_alerts})', fontsize=14)
        else:
            ax2.text(0.5, 0.5, 'No alerts generated', ha='center', va='center')
            ax2.set_title('Alert Accuracy', fontsize=14)
        
        plt.suptitle(f'Simulation Performance - {results["simulation_id"]}', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_vehicle_performance_chart(self, results: Dict, output_file: Path):
        """Create chart showing performance by vehicle."""
        vehicle_summaries = results.get('summary_by_vehicle', {})
        
        if not vehicle_summaries:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No vehicle data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Vehicle Performance')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            return
        
        # Prepare data
        vehicles = []
        shifts = []
        alerts = []
        avg_trips = []
        
        for vehicle_id, summary in vehicle_summaries.items():
            vehicles.append(f"Vehicle {vehicle_id}")
            shifts.append(summary.get('total_shifts', 0))
            alerts.append(summary.get('total_alerts', 0))
            avg_trips.append(summary.get('average_trips_completed', 0))
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Shifts and alerts
        x = range(len(vehicles))
        width = 0.35
        
        bars1 = ax1.bar([i - width/2 for i in x], shifts, width, label='Shifts', color='#3498db')
        bars2 = ax1.bar([i + width/2 for i in x], alerts, width, label='Alerts', color='#e74c3c')
        
        ax1.set_xlabel('Vehicle', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('Shifts and Alerts by Vehicle', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(vehicles, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Average trips completed
        bars3 = ax2.bar(x, avg_trips, color='#2ecc71')
        
        # Add target line
        target_trips = 4  # From config
        ax2.axhline(y=target_trips, color='#e74c3c', linestyle='--', 
                   label=f'Target ({target_trips} trips)')
        
        ax2.set_xlabel('Vehicle', fontsize=12)
        ax2.set_ylabel('Average Trips per Shift', fontsize=12)
        ax2.set_title('Average Trip Completion by Vehicle', fontsize=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels(vehicles, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Vehicle Performance Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_confusion_matrix_chart(self, results: Dict, output_file: Path):
        """Create confusion matrix visualization."""
        metrics = results.get('metrics', {})
        confusion = metrics.get('confusion_matrix', {})
        
        # Create confusion matrix
        matrix = [
            [confusion.get('true_negatives', 0), confusion.get('false_positives', 0)],
            [confusion.get('false_negatives', 0), confusion.get('true_positives', 0)]
        ]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Alert\n(Predicted)', 'Alert\n(Predicted)'],
                   yticklabels=['Success\n(Actual)', 'Problem\n(Actual)'],
                   ax=ax, cbar_kws={'label': 'Count'})
        
        ax.set_title('Confusion Matrix - Alert Predictions', fontsize=16)
        
        # Add text annotations
        total = sum(sum(row) for row in matrix)
        if total > 0:
            accuracy = (matrix[0][0] + matrix[1][1]) / total
            ax.text(0.5, -0.15, f'Overall Accuracy: {accuracy:.1%}', 
                   ha='center', transform=ax.transAxes, fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_trip_completion_chart(self, timeline_df: pd.DataFrame, output_file: Path):
        """Create chart showing trip completion over time."""
        if timeline_df.empty:
            return
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Convert to local timezone
        timeline_df['timestamp_local'] = timeline_df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(self.tz)
        
        # Sample vehicles to show (top 5 by number of points)
        top_vehicles = timeline_df['vehicle_id'].value_counts().head(5).index
        
        # Plot lines for each vehicle
        for vehicle_id in top_vehicles:
            vehicle_data = timeline_df[timeline_df['vehicle_id'] == vehicle_id].copy()
            vehicle_data = vehicle_data.sort_values('timestamp_local')
            
            ax.plot(vehicle_data['timestamp_local'], 
                   vehicle_data['trips_completed'],
                   marker='o', markersize=4, 
                   label=f'Vehicle {vehicle_id}')
        
        # Add target line
        ax.axhline(y=4, color='red', linestyle='--', alpha=0.7, label='Target (4 trips)')
        
        ax.set_xlabel(f'Time ({self.timezone})', fontsize=12)
        ax.set_ylabel('Trips Completed', fontsize=12)
        ax.set_title('Trip Completion Progress Over Time', fontsize=16)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M', tz=self.tz))
        plt.xticks(rotation=45, ha='right')
        
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()