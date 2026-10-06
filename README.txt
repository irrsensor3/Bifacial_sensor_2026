Bifacial Sensor 2026

Bifacial sensor monitoring and analysis system for rooftop photovoltaic research and field monitoring. The project combines a Raspberry Pi data logger, cloud storage, and a Streamlit dashboard to collect irradiance data, monitor panel performance, identify anomalies, and generate reports.

Overview

This repository contains the core software used to:
- read bifacial irradiance sensors from multiple I2C buses on a Raspberry Pi
- log data locally to CSV files and push readings to Supabase
- display live monitoring, panel performance, and irradiance trends in a browser dashboard
- detect abnormal readings and data quality issues
- generate data summaries and export reports in DOCX and PDF formats

The system is designed for a practical field deployment where sensor hardware may be partially connected, readings may be noisy, and the dashboard must continue operating without failing if a single channel or upstream service is temporarily unavailable.

Project scope

The repository includes both the field-side data acquisition layer and the analysis/visualisation layer:
- bifacial_logger.py: sensor logger for Raspberry Pi hardware
- app.py: main Streamlit dashboard entry point
- ui_sections.py: shared UI components, theming, data fetching, and plotting helpers
- Live_Monitoring.py: real-time monitoring views
- Panel_Array.py: panel-level output views
- Irradiance_Tracker.py: irradiance performance and timeline analytics
- Data_and_Reports.py: data inspection and report generation
- Anomalies.py: anomaly detection and fault review
- Admin_Controls.py: admin settings and operational controls
- detector.py, Gap_Filling.py, pv_gapfill.py: data-quality and gap-filling routines
- drive_fetch.py: Google Drive data access utilities
- nightly_check.py: scheduled validation / maintenance logic

Hardware and data flow

The logger reads 24 irradiance/temperature channels distributed across three I2C buses on the Raspberry Pi. Each bus uses ADS1115 boards with multiple analog input channels. The logger samples irradiance on a short interval and records temperature once per minute, along with running averages for each sensor.

Data is then written to local CSV files and optionally pushed to Supabase. The dashboard reads the cloud data to display the current state of the array, detect issues, and allow report generation.

System architecture

1. Raspberry Pi field node
   - reads sensors through three I2C buses
   - validates readings for obvious faults
   - writes per-day CSV files
   - optionally pushes data to Supabase

2. Supabase backend
   - stores sensor readings, alerts, admin settings, and control data
   - serves as the live source for the dashboard

3. Streamlit web app
   - live monitoring
   - panel performance visualisation
   - anomaly review
   - reporting and export
   - admin controls

Repository layout

app.py
Main Streamlit application entry point and navigation hub.

ui_sections.py
Shared dashboard styling, login flow, helper functions, plots, and data fetch logic.

Live_Monitoring.py
Live sensor views and monitoring summary screens.

Panel_Array.py
Panel-level visualisation and array output views.

Irradiance_Tracker.py
Irradiance trends, direct beam comparisons, and tracker views.

Data_and_Reports.py
Report generation and dataset review screens.

Anomalies.py
Anomaly identification and analysis pages.

Admin_Controls.py
Administrative controls and operational toggles.

bifacial_logger.py
Field logger that reads sensors, stores data, and handles Supabase sync.

detector.py
Additional detection logic for signal quality and data validation.

Gap_Filling.py
Gap-filling and imputation routines for incomplete sensor data.

pv_gapfill.py
PV-specific gap-filling logic used in data quality workflows.

drive_fetch.py
Google Drive integration utilities for retrieving external files.

nightly_check.py
Nightly maintenance or validation script for operational checks.

requirements.txt
Python dependency list for the project.

README.txt
Project documentation.

Requirements

Python 3.10 or newer recommended

Core Python packages:
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

1. Create and activate a Python environment.
2. Install the requirements.
3. Configure the required Supabase credentials.
4. Start the dashboard:

streamlit run app.py

The app will start on the default Streamlit port, typically:

http://localhost:8501

Environment and configuration

The dashboard expects configuration values for this project, especially the Supabase connection details. These should not be committed directly to the repository.

Typical setup options:
- Streamlit secrets file: .streamlit/secrets.toml
- environment variables loaded during startup
- secure host-specific configuration in a deployment environment

Example secrets file:

SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-supabase-key"

Security notes:
- keep credentials out of source control
- avoid exposing service-role keys in client-facing code
- use role-based access and restrict database permissions where possible
- verify local and remote config before production deployment

Running the field logger

The logger script is intended for a Raspberry Pi connected to the configured sensor hardware.

Example:

python bifacial_logger.py

The logger will:
- open the configured I2C buses
- read the sensor channels
- validate readings
- write CSV output to the local data directory
- optionally push readings to Supabase

The hardware configuration is defined in the script and expects the Raspberry Pi I2C overlays to be enabled and the ADS1115 boards to be present on the expected addresses.

Data storage

The logger stores data in a local directory structure based on year and month, with per-day CSV files. A typical layout is:

~/Desktop/bifacial data/
  2026/
    10/
      Bifacial_2026-10-06.csv

Each row contains date, time, irradiance values, temperature values, and rolling irradiance averages for each configured sensor.

Dashboard features

The Streamlit application provides a practical operational dashboard for the field installation:
- live monitoring of current irradiance readings
- panel-by-panel output review
- front and rear irradiance comparison
- anomaly detection and flagged issues
- data reports for review and export
- admin controls for system operation and sensor configuration

Admin access

The application includes a login flow and admin role handling. In a production deployment, the authentication method should be reviewed and hardened against a simple local credential model.

Recommended operational practice:
- keep admin credentials outside code
- restrict administrative access to trusted users
- review any force-log or override behaviour before use in production

Reports and exports

The repo includes report generation utilities for professional export outputs:
- DOCX report generation
- PDF report generation
- embedded plots and summary visuals

This is useful for site reporting, daily review, and project documentation.

Deployment notes

The application can be run locally or deployed to a server environment.

Typical deployment approaches:
- local workstation for development and test
- Raspberry Pi + local dashboard access in the field
- remote server running the Streamlit dashboard with cloud database connectivity
- container-based deployment if required

For headless environments, ensure the plotting backend is configured correctly for Matplotlib, especially when generating reports or running in Docker.

Troubleshooting

Common issues:
- missing Supabase credentials
- wrong I2C bus configuration on the Raspberry Pi
- ADS1115 boards not detected on expected addresses
- sensors returning negative or unrealistic values
- missing or stale local data files
- Matplotlib backend issues in headless environments

In such cases, check the logs generated by the Python scripts and verify the sensor map, bus configuration, and database connectivity.

Project status

This repository is a field monitoring and analysis project for bifacial PV research. It is structured around operational monitoring and reporting rather than a generic template project. The codebase includes both hardware integration code and analytical dashboard tools.

License

No explicit license file is present in the repository at this time. If this project is intended for public distribution, add a LICENSE file before publishing or sharing it more widely.

Contact

Repository owner: irrsensor3

This project is intended for the operation, monitoring, and analysis of the bifacial sensor installation described in the repository.
