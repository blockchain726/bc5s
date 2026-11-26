"""Read-only display of the independent paper lab; never controls trading."""
from __future__ import annotations

from datetime import datetime,timezone
import html
import json
import math
from pathlib import Path
import time


def read_status(log_dir,now=None):
    path=Path(log_dir)/"model_lab/status.json"
    try:
        if path.stat().st_size>2_000_000:
            raise ValueError("oversized research status")
        d=json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(d,dict):
            raise ValueError("invalid research status")
        updated=datetime.fromisoformat(d["updated_utc"].replace("Z","+00:00")).timestamp()
        age=max(0.,(time.time() if now is None else now)-updated)
        return {**d,"available":True,"fresh":age<=45,"age_s":round(age,1)}
    except (OSError,ValueError,KeyError,TypeError):
        return {"available":False,"fresh":False,"models":[],"results":{},
                "message":"Market Memory paper research is not available yet"}


def number(v):
    return float(v) if isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) else None


def panel_card(d):
    """Compact card for the existing BC page sidebar."""
    e=lambda s:html.escape(str(s),quote=True)
    head='<div class="c"><div class="ct"><span>MARKET MEMORY</span><span class="mute">paper research</span></div>'
    if not d.get("available"):
        return head+'<div class="mute">Waiting for the paper research service</div></div>'
    running="Running" if d.get("fresh") else "Not updating"
    out=[head,f'<div class="two mute"><span>{running}</span><span>win rate &middot; bets</span></div>']
    models=d.get("models") or []
    for i,m in enumerate(models):
        result=(d.get("results") or {}).get(m.get("id"),{}).get("selected",{})
        n=number(result.get("calls")) or 0
        ok=number(result.get("wins")) or 0
        pnl=number(result.get("net_flat_1"))
        label="Current" if m.get("role")=="incumbent" else f"Challenger {i}"
        record=f"{ok/n:.1%} &middot; {int(n)}" if n else "&ndash;"
        out.append(f'<div class="two" title="{e(m.get("id",""))}"><span>{label}</span><b>{record}</b></div>')
        if pnl is not None:
            out.append(f'<div class="two mute"><span>flat $1 P&amp;L</span><span>{pnl:+.2f}</span></div>')
    cadence=number(d.get("refit_hours")) or 6
    msg="Training a challenger" if d.get("training_pid") else f"Retrains every {cadence:g} h"
    out.append(f'<div class="mute">{msg} &middot; records kept across refits</div></div>')
    return "".join(out)
