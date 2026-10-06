Bifacial Sensor 2026

Operational monitoring platform for bifacial photovoltaic performance and field sensor data.

Overview

Bifacial Sensor 2026 is a field monitoring system designed to collect, validate, monitor, and report bifacial irradiance and panel performance data from a rooftop PV installation. The platform combines a Raspberry Pi-based sensor logger, cloud data storage, and a Streamlit operations dashboard to provide live visibility into system performance and data quality.

The system is built for day-to-day operational use. It is intended to help operators answer practical questions such as:
- Are the sensors reporting normally?
- Are panel outputs consistent with expected irradiance?
- Is a sensor channel failing or disconnected?
- Are there abnormal conditions requiring investigation?
- Can reports be generated quickly from current and historical data?

Product scope

This repository contains the complete monitoring and analysis stack for the project:
- Raspberry Pi logger for sensor acquisition and local data capture
- data validation and anomaly handling
- cloud synchronization through Supabase
- operational dashboard for live monitoring
- panel and irradiance analytics
- reporting and export tools
- admin and system control functions

System architecture

The system follows a simple operational workflow:

1. Sensor acquisition
   The Raspberry Pi reads irradiance and temperature data from multiple sensor channels distributed across several I2C buses.

2. Local data persistence
   Readings are written to local CSV files for durable storage and historical recovery.

3. Cloud synchronization
   Valid data is pushed to Supabase, providing a central live source for the dashboard and downstream analysis.

4. Operational dashboard
   The Streamlit application surfaces live readings, trends, panel output, anomalies, and reporting views.

5. Operational response
   Admin tools and alert logic support maintenance actions, manual overrides, and system health checks.

Repository structure

app.py
Main Streamlit application entry point and navigation layer.

ui_sections.py
Shared dashboard styling, authentication, helper utilities, fetching logic, charts, and report support.

Live_Monitoring.py
Live monitoring views and current operating status.

Panel_Array.py
Panel-level output and array-level performance views.

Irradiance_Tracker.py
Irradiance trend analysis and tracking views.

Data_and_Reports.py
Dataset review and report generation functionality.

Anomalies.py
Anomaly detection and issue review workflow.

Admin_Controls.py
Admin settings and system controls.

bifacial_logger.py
Field logger responsible for hardware sampling, local CSV logging, and Supabase synchronization.

detector.py
Detection and validation logic for signal quality and abnormal conditions.

Gap_Filling.py
Gap-filling logic for incomplete or interrupted readings.

pv_gapfill.py
PV-specific data quality and filling routines.

drive_fetch.py
Google Drive integration utilities for retrieving external files or support data.

nightly_check.py
Scheduled operational checks or maintenance routines.

requirements.txt
Python dependencies for the project.

README.txt
Repository documentation and operations reference.

What the platform does

Live monitoring
- view current irradiance conditions
- monitor panel output in near real time
- track front and rear irradiance differences
- review array status across multiple panels

Data quality management
- detect missing or invalid sensor channels
- treat grounded or disconnected channels safely
- log anomalies and alerts
- support data gap handling and repair workflows

Reporting
- generate operational summaries
- produce DOCX and PDF exports
- package reports with charts and summary metrics

Administration
- control access through a login flow
- support admin-only operational controls
- enable sensor configuration and override actions where required

Operations and deployment model

This project is designed for real field operations rather than a generic demo application. It supports a practical workflow in which:
- sensor hardware sits on-site and logs continuously
- the Pi stores raw data locally for resilience
- the dashboard provides live situational awareness to operators
- reporting can be generated with minimal delay
- cloud-backed data access reduces dependence on direct local access

Typical operational setup

- Raspberry Pi field node connected to the sensor array
- multiple I2C buses and ADS1115 boards handling sensor acquisition
- local CSV data storage on the Pi
- Supabase used as the central data and control layer for the dashboard
- Streamlit dashboard served for operational review and reporting

Requirements

Recommended runtime:
- Python 3.10+
- Linux or Raspberry Pi OS for the logger
- Windows, Linux, or macOS for local dashboard development

Core dependencies:
- streamlit
- pandas
- numpy
- matplotlib
- plotly
- scikit-learn
- python-docx
- fpdf2
- supabase
- google-api-python-client
- google-auth

Install dependencies:

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

On Windows PowerShell:

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Quick start

Run the web application:

streamlit run app.py

Start the field logger on the Raspberry Pi:

python bifacial_logger.py

The dashboard is typically served on:

http://localhost:8501

Configuration

The project depends on external configuration for sensor access and cloud connectivity. Store credentials securely and do not commit them to the repository.

Recommended configuration sources:
- Streamlit secrets file
- environment variables
- deployment platform secret storage

Example `.streamlit/secrets.toml`:

SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-supabase-key"

Security guidance:
- keep production secrets out of source control
- avoid exposing service-role credentials in client-facing code
- use restricted database permissions and role-based access where possible
- validate access before deploying to a public or shared environment

Operational notes

The logger is designed to handle real-world field conditions where sensors may be missing, partially connected, or temporarily unstable. The system is deliberately resilient in these cases:
- invalid readings are not allowed to crash the logger
- missing boards are retried without blocking the main workflow
- disconnected sensor channels are treated safely and logged appropriately
- cloud outages do not stop local logging from continuing

Data output

The logger writes daily CSV files under a year/month directory structure, for example:

~/Desktop/bifacial data/
  2026/
    10/
      Bifacial_2026-10-06.csv

The output includes date, time, irradiance values, temperature values, and per-sensor rolling averages.

Operational dashboard features

The dashboard is designed for day-to-day operational review:
- real-time status of the array and sensors
- live irradiance and panel power views
- panel-by-panel output breakdown
- front and rear irradiance comparison
- anomaly detection and flagged conditions
- export-ready reporting for site review
- admin controls for operational configuration and maintenance

Admin access

The repository contains an authentication flow and admin-specific controls. This should be treated as operational access and hardened before production deployment. In a live environment, administrative access should be restricted to approved users and protected by secure credentials.

Troubleshooting

Common operational issues:
- missing Supabase credentials
- unavailable or misconfigured I2C buses
- ADS1115 boards not detected at expected addresses
- unrealistic or negative sensor values
- stale or incomplete local data files
- Matplotlib backend problems in headless environments

Recommended checks:
- verify the Pi I2C overlay configuration
- confirm bus and ADS1115 address mapping
- check the sensor wiring and channel mapping
- inspect logger output for warnings and connection failures
- validate Supabase connectivity and row access

Reporting and document output

The project includes support for generating operational documents and summary reports:
- DOCX reports
- PDF exports
- chart-based summary visuals

This is useful for site reviews, maintenance reporting, and project documentation.

Deployment guidance

This application can be deployed in multiple operational models:
- local development environment
- Raspberry Pi field deployment
- remote server for dashboard access
- container-based deployment for managed infrastructure

For production use, ensure the hosting environment is configured for reliable access to both the data source and the dashboard.

Security and governance

- keep all credentials outside the repository
- protect admin access and restrict it to trusted users
- avoid exposing service-role keys in browser-accessible code
- review user permissions for Supabase and related services
- validate operational actions before enabling them in production

Project status

This repository is a field monitoring and analytics platform for bifacial photovoltaic performance tracking. It is designed for operational visibility, data integrity, and reporting rather than as a generic reference project.

License

No explicit license file is currently present in the repository. If this project is intended for wider distribution or external use, add an appropriate license before publication.

Repository owner

irrsensor3

This project is intended for the operation, monitoring, and analysis of the bifacial sensor installation described in this repository.
