@echo off
cd /d "%~dp0"
start "Federated SNN Lab" "http://localhost:8501"
python -m streamlit run dashboard.py --server.headless false --server.port 8501
