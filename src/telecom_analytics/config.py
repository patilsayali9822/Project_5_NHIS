from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES = PROJECT_ROOT / "reports" / "figures"
MODELS = PROJECT_ROOT / "models"

USER_ID = "MSISDN/Number"
SESSION_ID = "Bearer Id"
DURATION = "Dur. (ms)"
HANDSET_MFR = "Handset Manufacturer"
HANDSET_TYPE = "Handset Type"

APP_PAIRS = {
    "social_media": ("Social Media DL (Bytes)", "Social Media UL (Bytes)"),
    "google": ("Google DL (Bytes)", "Google UL (Bytes)"),
    "email": ("Email DL (Bytes)", "Email UL (Bytes)"),
    "youtube": ("Youtube DL (Bytes)", "Youtube UL (Bytes)"),
    "netflix": ("Netflix DL (Bytes)", "Netflix UL (Bytes)"),
    "gaming": ("Gaming DL (Bytes)", "Gaming UL (Bytes)"),
    "other": ("Other DL (Bytes)", "Other UL (Bytes)"),
}

EXPERIENCE_COLS = {
    "tcp_dl": "TCP DL Retrans. Vol (Bytes)",
    "tcp_ul": "TCP UL Retrans. Vol (Bytes)",
    "rtt_dl": "Avg RTT DL (ms)",
    "rtt_ul": "Avg RTT UL (ms)",
    "tp_dl": "Avg Bearer TP DL (kbps)",
    "tp_ul": "Avg Bearer TP UL (kbps)",
}
