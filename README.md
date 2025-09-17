# Booth Visitor Analytics (WebRTC + YOLOv8n)

App that detects & counts people per-zone **in the browser webcam** via WebRTC.
Logs go to CSVs. Zones saved to `zones.json`. Everything persists on Render using a disk.

## Local dev
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
