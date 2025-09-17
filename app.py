import os, json, csv, time, datetime, base64, threading
from io import BytesIO
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
from PIL import Image as PILImage
from ultralytics import YOLO

# Optional (nicer Data tab)
try:
    import pandas as pd
except Exception:
    pd = None

# WebRTC
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av

# ------------------------- Paths & persistence (Render-friendly) -------------------------
BASE_DIR = Path(os.getenv("LOG_DIR", "."))  # on Render: /var/data
BASE_DIR.mkdir(parents=True, exist_ok=True)

# Persist YOLO cache to disk too (faster cold start on Render)
os.environ.setdefault("ULTRALYTICS_CACHE_DIR", str(BASE_DIR / "ultralytics_cache"))
(Path(os.environ["ULTRALYTICS_CACHE_DIR"])).mkdir(parents=True, exist_ok=True)

BASE_LOG_DIR = BASE_DIR / "logs"
BASE_LOG_DIR.mkdir(parents=True, exist_ok=True)

SNAPSHOT_PATH = BASE_DIR / "snapshot.jpg"
ZONES_PATH = BASE_DIR / "zones.json"
LATEST_CSV_PATH = BASE_LOG_DIR / "visitor_log.csv"

# ------------------------- Global YOLO (load once) -------------------------
_YOLO_MODEL = None
def get_model():
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        _YOLO_MODEL = YOLO("yolov8n.pt")
    return _YOLO_MODEL

# ------------------------- Lightweight SORT tracker ------------------------
class Track:
    def __init__(self, tid, bbox):
        self.id = tid; self.bbox = bbox; self.hits = 0; self.no_losses = 0

class Sort:
    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age; self.min_hits = min_hits
        self.iou_threshold = iou_threshold; self.tracks = []; self.track_id_count = 0
    @staticmethod
    def _iou(a, b):
        xx1 = max(a[0], b[0]); yy1 = max(a[1], b[1])
        xx2 = min(a[2], b[2]); yy2 = min(a[3], b[3])
        w = max(0, xx2-xx1); h = max(0, yy2-yy1)
        inter = w*h
        area_a = max(0,(a[2]-a[0]))*max(0,(a[3]-a[1]))
        area_b = max(0,(b[2]-b[0]))*max(0,(b[3]-b[1]))
        denom = area_a + area_b - inter
        return inter/denom if denom>0 else 0.0
    def update(self, dets):
        if not dets:
            for t in self.tracks: t.no_losses += 1
            self.tracks = [t for t in self.tracks if t.no_losses <= self.max_age]
            return []
        updated=[]
        for det in dets:
            matched=False
            for trk in self.tracks:
                if self._iou(det,trk.bbox) >= self.iou_threshold:
                    trk.bbox=det; trk.hits+=1; trk.no_losses=0; updated.append(trk); matched=True; break
            if not matched:
                self.track_id_count+=1
                trk=Track(self.track_id_count, det)
                self.tracks.append(trk); updated.append(trk)
        for trk in self.tracks:
            if trk not in updated: trk.no_losses+=1
        self.tracks = [t for t in self.tracks if t.no_losses <= self.max_age]
        return [(t.id, t.bbox) for t in self.tracks if t.hits >= self.min_hits]

# ------------------------- Zones & geometry helpers ------------------------
def save_zones_payload(canvas_w, canvas_h, zones_list):
    with open(ZONES_PATH, "w") as f:
        json.dump({"base_width":int(canvas_w), "base_height":int(canvas_h), "zones":zones_list}, f, indent=2)

def load_zones_payload():
    if not ZONES_PATH.exists(): return None
    with open(ZONES_PATH) as f: data=json.load(f)
    if isinstance(data, list): return {"base_width":None,"base_height":None,"zones":data}
    return data

def scale_zones_to_frame(zp, frame_w, frame_h):
    bw, bh = zp["base_width"], zp["base_height"]; zones=zp["zones"]
    if not bw or not bh or bw<=0 or bh<=0: return zones
    sx, sy = frame_w/bw, frame_h/bh
    return [{"name":z["name"],
             "x1":int(round(z["x1"]*sx)),"y1":int(round(z["y1"]*sy)),
             "x2":int(round(z["x2"]*sx)),"y2":int(round(z["y2"]*sy))} for z in zones]

def point_in_zone(cx, cy, zones):
    for z in zones:
        if z["x1"]<=cx<=z["x2"] and z["y1"]<=cy<=z["y2"]: return z["name"]
    return "None"

def pil_to_data_url(img: PILImage, width=None, fmt="JPEG"):
    if width and width>0:
        w = int(width); h = int(round(img.height * (w / img.width)))
        img = img.resize((w, h))
    buf = BytesIO(); img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"

# ------------------------- Minimal HTML zone mapper ------------------------
def zone_mapper_html(data_url: str, width: int, height: int) -> str:
    return f"""
<div style="font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial;margin-bottom:6px;">
  <button id="undoBtn">Undo</button>
  <button id="clearBtn">Clear</button>
  <button id="copyBtn">Copy JSON</button>
  <a id="dl" download="zones_rects.json" style="margin-left:8px;">Download JSON</a>
  <span style="margin-left:8px;opacity:.7;">Tip: click–drag to draw zones</span>
</div>
<canvas id="zcanvas" width="{width}" height="{height}" style="border:1px solid #ddd; touch-action:none;"></canvas>
<script>
const canvas = document.getElementById('zcanvas');
const ctx = canvas.getContext('2d');
const bg = new Image(); bg.src = "{data_url}";
let rects = [];
let drawing = false, sx=0, sy=0, cx=0, cy=0;

function drawAll(preview=false) {{
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(bg, 0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 2; ctx.strokeStyle = '#ff0000'; ctx.fillStyle = 'rgba(0,0,255,0.18)';
  for (const r of rects) {{
    ctx.fillRect(r.x1, r.y1, r.x2 - r.x1, r.y2 - r.y1);
    ctx.strokeRect(r.x1, r.y1, r.x2 - r.x1, r.y2 - r.y1);
  }}
  if (preview) {{
    const x1 = Math.min(sx, cx), y1 = Math.min(sy, cy);
    const x2 = Math.max(sx, cx), y2 = Math.max(sy, cy);
    ctx.fillRect(x1, y1, x2-x1, y2-y1);
    ctx.strokeRect(x1, y1, x2-x1, y2-y1);
  }}
}}
function toJSON() {{ return JSON.stringify({{rects}}); }}
function pos(e) {{ const r = canvas.getBoundingClientRect(); return {{ x: Math.round(e.clientX - r.left), y: Math.round(e.clientY - r.top) }}; }}

canvas.addEventListener('mousedown', e => {{ drawing=true; const p=pos(e); sx=cx=p.x; sy=cy=p.y; drawAll(true); }});
canvas.addEventListener('mousemove', e => {{ if(!drawing) return; const p=pos(e); cx=p.x; cy=p.y; drawAll(true); }});
canvas.addEventListener('mouseup', e => {{
  if(!drawing) return; drawing=false;
  const x1=Math.min(sx,cx), y1=Math.min(sy,cy), x2=Math.max(sx,cx), y2=Math.max(sy,cy);
  if((x2-x1)>4 && (y2-y1)>4) rects.push({{x1,y1,x2,y2}});
  drawAll(false);
}});
canvas.addEventListener('mouseleave', e => {{ if(drawing) {{ drawing=false; drawAll(false); }} }});
document.getElementById('undoBtn').onclick=()=>{{ rects.pop(); drawAll(false); }};
document.getElementById('clearBtn').onclick=()=>{{ rects=[]; drawAll(false); }};
document.getElementById('copyBtn').onclick=async()=>{{ try{{ await navigator.clipboard.writeText(toJSON()); alert('Copied! Paste into the box below.'); }}catch(e){{ alert('Copy failed, use Download'); }} }};
document.getElementById('dl').onclick=(e)=>{{ const blob=new Blob([toJSON()], {{type:'application/json'}}); e.target.href=URL.createObjectURL(blob); }};
bg.onload=()=>drawAll(false);
</script>
"""

# ------------------------- Logging & shared store -------------------------
def ensure_logs_dir():
    BASE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    return str(BASE_LOG_DIR)

def new_session_id():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def write_header(path, overwrite=False):
    mode = "w" if overwrite else ("a" if os.path.exists(path) else "w")
    with open(path, mode, newline="") as f:
        w = csv.writer(f)
        if mode == "w":
            w.writerow(["session_id","timestamp","visitor_id","x1","y1","x2","y2","cx","cy","zone"])

def append_row(path, session_id, timestamp, tid, x1,y1,x2,y2,cx,cy,zone):
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([session_id, timestamp, tid, x1, y1, x2, y2, cx, cy, zone])

class SharedStore:
    def __init__(self):
        self.lock = threading.Lock()
        self.session_id = ""
        self.log_path = str(LATEST_CSV_PATH)  # default
        self.latest_path = str(LATEST_CSV_PATH)
        self.zones_payload = None
        self.zones_scaled = None
        self.frame_size = None  # (w,h)
        self.visitors = {}         # tid -> dict
        self.counts = {}           # zone -> active count
        self.latest_frame = None   # last BGR frame (numpy)

SHARED = SharedStore()

def make_rtc_configuration():
    ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
    turn_urls = os.getenv("TURN_URLS", "").strip()
    if turn_urls:
        urls = [u.strip() for u in turn_urls.split(",") if u.strip()]
        ice_servers.append({
            "urls": urls,
            "username": os.getenv("TURN_USERNAME",""),
            "credential": os.getenv("TURN_PASSWORD","")
        })
    return {"iceServers": ice_servers}

# ------------------------- Streamlit app -------------------------
st.set_page_config(layout="wide")
st.title("👥 Booth Visitor Analytics POC (WebRTC-only)")

tab1, tab2, tab3 = st.tabs(["📍 Zone Mapper", "📊 Analytics", "📈 Data"])

# Session state (minimal)
ss = st.session_state
DEFAULTS = {
    "snapshot_path": str(SNAPSHOT_PATH) if SNAPSHOT_PATH.exists() else None,
    "canvas_w": 960,
    "canvas_token": 0,
    "running": False,
    "session_id": None,
    "video_mode": "Browser Webcam (WebRTC)",
    "zones_payload": None,
    "zones_scaled": None,
    "latest_path": str(LATEST_CSV_PATH),
}
for k, v in DEFAULTS.items():
    if k not in ss: ss[k] = v

GRACE_SEC = 1.5

# ------------------------- TAB 1: Zone Mapper -------------------------
with tab1:
    st.subheader("Draw Zones on a Snapshot")

    cam_photo = st.camera_input(
        "Take a photo (browser webcam)",
        help="Use this to capture a still snapshot at good quality."
    )
    if cam_photo is not None:
        img = PILImage.open(cam_photo).convert("RGB")
        img.save(SNAPSHOT_PATH, format="JPEG", quality=95)
        ss["snapshot_path"] = str(SNAPSHOT_PATH)
        ss["canvas_token"] += 1
        st.success(f"Snapshot saved to {SNAPSHOT_PATH}")

    if st.button("Use current frame from live Analytics stream"):
        with SHARED.lock:
            frame = None if SHARED.latest_frame is None else SHARED.latest_frame.copy()
        if frame is None:
            st.warning("No live frame yet. Start Analytics first.")
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            PILImage.fromarray(rgb).save(SNAPSHOT_PATH, format="JPEG", quality=95)
            ss["snapshot_path"] = str(SNAPSHOT_PATH)
            ss["canvas_token"] += 1
            st.success(f"Snapshot grabbed → {SNAPSHOT_PATH}")

    up = st.file_uploader("…or upload a still image", type=["jpg","jpeg","png"])
    if up:
        with open(SNAPSHOT_PATH, "wb") as f: f.write(up.read())
        ss["snapshot_path"]=str(SNAPSHOT_PATH); ss["canvas_token"] += 1
        st.success(f"Snapshot loaded → {SNAPSHOT_PATH}")

    rects = []
    if ss["snapshot_path"] and Path(ss["snapshot_path"]).exists():
        try:
            pil_img = PILImage.open(ss["snapshot_path"]).convert("RGB")
        except Exception:
            st.error("Failed to read snapshot file."); pil_img=None

        if pil_img:
            max_w = min(1600, pil_img.width)
            ss["canvas_w"] = st.slider("Canvas width", 600, max_w, value=min(960, max_w), step=20)
            disp_w = int(ss["canvas_w"])
            disp_h = int(round(pil_img.height * (disp_w / pil_img.width)))
            data_url = pil_to_data_url(pil_img, width=disp_w, fmt="JPEG")

            components.html(zone_mapper_html(data_url, disp_w, disp_h), height=disp_h+80, scrolling=False)

            st.markdown("**Paste rects JSON here (after clicking ‘Copy JSON’ above):**")
            rects_text = st.text_area("Rects JSON", value="", height=120, label_visibility="collapsed")
            if rects_text.strip():
                try:
                    rects_raw = json.loads(rects_text)
                    rects = rects_raw.get("rects", []) if isinstance(rects_raw, dict) else rects_raw
                except Exception as e:
                    st.error(f"Invalid JSON: {e}")
                    rects = []

            zones=[]
            if rects:
                st.write("Name your zones:")
                for i, r in enumerate(rects):
                    name = st.text_input(f"Zone {i+1} name", f"Zone{i+1}")
                    zones.append({"x1":int(r["x1"]), "y1":int(r["y1"]), "x2":int(r["x2"]), "y2":int(r["y2"]), "name":name})

            c1, c2 = st.columns(2)
            if c1.button("💾 Save Zones", disabled=(len(zones)==0)):
                save_zones_payload(disp_w, disp_h, zones)
                st.success(f"Zones saved → {ZONES_PATH}")
            if c2.button("🗑️ Clear Saved Zones"):
                save_zones_payload(disp_w, disp_h, [])
                st.warning("zones.json cleared (kept base dimensions).")

# ------------------------- TAB 2: Analytics (WebRTC only) -------------------------
with tab2:
    st.subheader("Run Analytics (Browser Webcam)")

    # Video quality controls
    res_label = st.selectbox("Resolution", ["1280×720", "1920×1080", "640×480"], index=0,
                             help="Higher = better quality, more CPU/bandwidth.")
    fps = st.selectbox("Frame rate", [15, 24, 30], index=2)

    W, H = (1280,720)
    if "1920" in res_label: W,H = (1920,1080)
    if "640" in res_label:  W,H = (640,480)

    conf_th = st.slider("YOLO confidence", 0.2, 0.9, 0.60, 0.01)
    iou_nms = st.slider("YOLO IoU NMS", 0.1, 0.9, 0.45, 0.01)

    zones_payload = load_zones_payload()
    if not zones_payload or not zones_payload.get("zones"):
        st.warning("No zones defined yet. Use the Zone Mapper tab.")
    else:
        st.caption(f"{len(zones_payload['zones'])} zones loaded.")

    start_clicked = st.button("▶️ Start Analytics", disabled=ss["running"])
    stop_clicked  = st.button("⏹️ Stop Analytics", disabled=not ss["running"])

    class BrowserCamProcessor(VideoTransformerBase):
        def __init__(self):
            self.model = get_model()
            self.tracker = Sort()
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            h,w = img.shape[:2]

            with SHARED.lock:
                # (Re)scale zones if payload changed or frame size changed
                if (SHARED.zones_payload is not zones_payload) or (SHARED.frame_size != (w,h)) or (SHARED.zones_scaled is None):
                    SHARED.zones_payload = zones_payload
                    SHARED.zones_scaled = scale_zones_to_frame(zones_payload, w, h) if zones_payload else []
                    SHARED.frame_size = (w,h)

            results = self.model.predict(img, conf=conf_th, iou=iou_nms, verbose=False, imgsz=640)[0]
            dets=[]
            for box in results.boxes:
                if int(box.cls[0])==0:
                    x1,y1,x2,y2=map(int, box.xyxy[0]); dets.append([x1,y1,x2,y2])

            tracked = self.tracker.update(dets)
            now_iso = datetime.datetime.now().isoformat()
            now_t = time.time()

            with SHARED.lock:
                counts = {z["name"]:0 for z in (SHARED.zones_scaled or [])}
                for (tid,(x1,y1,x2,y2)) in tracked:
                    cx,cy=(x1+x2)//2,(y1+y2)//2
                    zone = point_in_zone(cx,cy,SHARED.zones_scaled or [])
                    # draw
                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.putText(img,f"ID {tid}",(x1,y1-6),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
                    # log
                    append_row(SHARED.log_path, SHARED.session_id, now_iso, tid, x1,y1,x2,y2,cx,cy,zone)
                    append_row(SHARED.latest_path, SHARED.session_id, now_iso, tid, x1,y1,x2,y2,cx,cy,zone)
                    # dwell
                    v = SHARED.visitors.get(tid)
                    if v is None:
                        v = {"first_seen": now_t, "last_seen": now_t, "current_zone": None, "entered_at": None, "zone_dwell": {}}
                        SHARED.visitors[tid]=v
                    v["last_seen"] = now_t
                    if v["current_zone"] != zone:
                        if v["current_zone"] and v["entered_at"]:
                            v["zone_dwell"][v["current_zone"]] = v["zone_dwell"].get(v["current_zone"],0.0)+(now_t-v["entered_at"])
                        v["current_zone"] = zone if zone!="None" else None
                        v["entered_at"] = now_t if v["current_zone"] else None
                # live counts
                for tid,v in SHARED.visitors.items():
                    cz=v.get("current_zone")
                    if cz and (now_t - v.get("last_seen",0) <= GRACE_SEC):
                        counts[cz]=counts.get(cz,0)+1
                SHARED.counts = counts
                SHARED.latest_frame = img.copy()

            # draw zones last
            zs = SHARED.zones_scaled or []
            for z in zs:
                cv2.rectangle(img,(z["x1"],z["y1"]),(z["x2"],z["y2"]),(255,0,0),2)
                label=f'{z["name"]} [{SHARED.counts.get(z["name"],0)}]'
                cv2.putText(img,label,(z["x1"],max(15,z["y1"]-6)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    if start_clicked and not ss["running"]:
        logs_dir = ensure_logs_dir()
        ss["session_id"] = new_session_id()
        ss["zones_payload"] = zones_payload
        ss["zones_scaled"] = None  # computed on first frame size

        with SHARED.lock:
            SHARED.session_id = ss["session_id"]
            SHARED.zones_payload = zones_payload
            SHARED.zones_scaled = None
            SHARED.frame_size = None
            SHARED.visitors = {}
            SHARED.log_path = str(BASE_LOG_DIR / f"{datetime.date.today()}_{ss['session_id']}.csv")
            SHARED.latest_path = str(LATEST_CSV_PATH)
            SHARED.latest_frame = None

        write_header(SHARED.log_path, overwrite=True)
        write_header(SHARED.latest_path, overwrite=True)
        ss["running"] = True

    if stop_clicked and ss["running"]:
        now_t = time.time()
        with SHARED.lock:
            for v in SHARED.visitors.values():
                if v.get("current_zone") and v.get("entered_at"):
                    v["zone_dwell"][v["current_zone"]] = v["zone_dwell"].get(v["current_zone"], 0.0) + (now_t - v["entered_at"])
                    v["entered_at"] = None
        ss["running"] = False
        st.success(f"Stopped session {ss['session_id']}")

    if ss["running"]:
        st.info("Browser webcam is active. The processed video (with overlays) appears below.")
        media_constraints = {
            "video": {
                "width":  {"ideal": int(W)},
                "height": {"ideal": int(H)},
                "frameRate": {"ideal": int(fps)},
                "facingMode": {"ideal": "environment"}
            },
            "audio": False
        }

        webrtc_streamer(
            key="webrtc-people-analytics",
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints=media_constraints,
            video_processor_factory=BrowserCamProcessor,
            async_processing=True,
            rtc_configuration=make_rtc_configuration(),
        )

        cols = st.columns(3)
        if cols[0].button("Refresh stats"):
            with SHARED.lock:
                active_now = sum(
                    1 for v in SHARED.visitors.values()
                    if v.get("current_zone") and (time.time()-v.get("last_seen",0)<=GRACE_SEC)
                )
                counts = dict(SHARED.counts)
            cols[1].markdown(f"**Session:** {ss['session_id']}")
            cols[2].markdown(f"**Active now:** {active_now}")
            if counts:
                st.write({k:int(v) for k,v in counts.items()})

# ------------------------- TAB 3: Data / Analytics -------------------------
with tab3:
    st.subheader("Explore Data")

    files = sorted([p for p in BASE_LOG_DIR.glob("*.csv")])
    latest_exists = Path(ss["latest_path"]).exists()

    if not files and not latest_exists:
        st.info("No data yet. Run Analytics to generate logs.")
        st.stop()

    def load_rows(paths):
        rows = []
        for p in paths:
            with open(p, newline="") as f:
                rd = csv.DictReader(f)
                for r in rd:
                    rows.append(r)
        return rows

    all_paths = files + ([Path(ss["latest_path"])] if latest_exists else [])

    # Date range defaults from filenames
    min_date = datetime.date.today(); max_date = datetime.date.today()
    if files:
        try:
            dates = []
            for p in files:
                s = p.name.split("_")[0]; y,m,d = s.split("-")
                dates.append(datetime.date(int(y), int(m), int(d)))
            if dates: min_date, max_date = min(dates), max(dates)
        except Exception:
            pass

    date_range = st.date_input("Date range", (min_date, max_date if max_date>=min_date else min_date))
    start_dt = datetime.datetime.combine(date_range[0], datetime.time.min)
    end_dt   = datetime.datetime.combine(date_range[1], datetime.time.max)

    zp = load_zones_payload()
    zone_names = [z["name"] for z in zp["zones"]] if (zp and zp.get("zones")) else []
    zones_filter = st.multiselect("Zones", options=zone_names, default=zone_names)
    min_dwell_min = st.slider("Min dwell per zone (min)", 0.0, 30.0, 0.0, 0.5)

    rows = load_rows(all_paths)
    if not rows:
        st.warning("No rows in logs yet."); st.stop()

    parsed = []
    for r in rows:
        try: ts = datetime.datetime.fromisoformat(r["timestamp"])
        except Exception: continue
        if not (start_dt <= ts <= end_dt): continue
        z = r.get("zone","None")
        if zones_filter and z not in zones_filter: continue
        try:
            parsed.append({
                "session": r.get("session_id",""),
                "timestamp": ts,
                "visitor_id": int(float(r.get("visitor_id", -1))),
                "x1": int(float(r.get("x1",0))), "y1": int(float(r.get("y1",0))),
                "x2": int(float(r.get("x2",0))), "y2": int(float(r.get("y2",0))),
                "cx": int(float(r.get("cx",0))), "cy": int(float(r.get("cy",0))),
                "zone": z
            })
        except Exception:
            pass

    if not parsed:
        st.info("No data in the selected range/filters."); st.stop()

    parsed.sort(key=lambda r: (r["session"], r["visitor_id"], r["timestamp"]))
    dwell_zone_secs = {}
    dwell_per_visitor_zone = {}
    unique_visitors = set()
    from collections import defaultdict
    dts_by_vid = defaultdict(list)

    prev_key = None; prev_zone = None; prev_time = None
    for r in parsed:
        key = (r["session"], r["visitor_id"])
        unique_visitors.add(key)
        if prev_key != key:
            if prev_key is not None and prev_zone is not None and prev_time is not None and dts_by_vid[prev_key]:
                approx = float(np.median(dts_by_vid[prev_key]))
                dwell_zone_secs[prev_zone] = dwell_zone_secs.get(prev_zone, 0.0) + approx
                dwell_per_visitor_zone[(prev_key[0], prev_key[1], prev_zone)] = dwell_per_visitor_zone.get((prev_key[0], prev_key[1], prev_zone), 0.0) + approx
            prev_key = key; prev_zone = r["zone"]; prev_time = r["timestamp"]; continue
        dt = (r["timestamp"] - prev_time).total_seconds()
        if 0 < dt < 60*5:
            dts_by_vid[key].append(dt)
            dwell_zone_secs[prev_zone] = dwell_zone_secs.get(prev_zone, 0.0) + dt
            dwell_per_visitor_zone[(key[0], key[1], prev_zone)] = dwell_per_visitor_zone.get((key[0], key[1], prev_zone), 0.0) + dt
        prev_zone = r["zone"]; prev_time = r["timestamp"]

    if prev_key is not None and prev_zone is not None and prev_time is not None and dts_by_vid[prev_key]:
        approx = float(np.median(dts_by_vid[prev_key]))
        dwell_zone_secs[prev_zone] = dwell_zone_secs.get(prev_zone, 0.0) + approx
        dwell_per_visitor_zone[(prev_key[0], prev_key[1], prev_zone)] = dwell_per_visitor_zone.get((prev_key[0], prev_key[1], prev_zone), 0.0) + approx

    by_minute = {}
    for r in parsed:
        minute = r["timestamp"].replace(second=0, microsecond=0)
        s = by_minute.setdefault(minute, set()); s.add((r["session"], r["visitor_id"]))
    timeline = sorted((m, len(s)) for m, s in by_minute.items())
    times = [t[0] for t in timeline]; counts = [t[1] for t in timeline]

    total_unique = len(unique_visitors)
    zone_summary = []
    for z in sorted(set([r["zone"] for r in parsed])):
        secs = dwell_zone_secs.get(z, 0.0)
        zone_summary.append({"zone": z, "dwell_min": round(secs/60.0, 2), "dwell_s": round(secs, 1)})
    zone_summary.sort(key=lambda x: -x["dwell_s"])

    pvz_rows = []
    for (sess, vid, z), secs in dwell_per_visitor_zone.items():
        if secs/60.0 >= min_dwell_min:
            pvz_rows.append({"session":sess, "visitor_id":vid, "zone":z, "dwell_min":round(secs/60.0,2), "dwell_s":round(secs,1)})
    pvz_rows.sort(key=lambda r: (-r["dwell_s"], r["session"], r["visitor_id"]))

    c1, c2, c3 = st.columns(3)
    c1.metric("Unique visitors", total_unique)
    c2.metric("Rows in view", len(parsed))
    c3.metric("Zones", len(zone_summary))

    st.markdown("#### Zone dwell (entry/exit based)")
    if pd:
        df_zone = pd.DataFrame(zone_summary)
        st.dataframe(df_zone, use_container_width=True)
        st.bar_chart(df_zone.set_index("zone")["dwell_min"])
    else:
        st.write(zone_summary)

    st.markdown("#### Timeline (unique visitors per minute)")
    if pd and len(times) > 0:
        df_time = pd.DataFrame({"minute": times, "visitors": counts}).set_index("minute")
        st.line_chart(df_time)
    else:
        st.write(list(zip(times, counts)))

    st.markdown(f"#### Visitors by zone (≥ {min_dwell_min} min dwell)")
    if pd and pvz_rows:
        df_pvz = pd.DataFrame(pvz_rows)
        st.dataframe(df_pvz, use_container_width=True, height=360)
    else:
        st.write(pvz_rows[:200])

    agg_path = BASE_LOG_DIR / "visitor_summary.csv"
    with open(agg_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["zone","dwell_s","dwell_min"])
        for row in zone_summary: w.writerow([row["zone"], row["dwell_s"], row["dwell_min"]])
    with open(agg_path, "rb") as f:
        st.download_button("Download zone summary (CSV)", f, "visitor_summary.csv", "text/csv")
