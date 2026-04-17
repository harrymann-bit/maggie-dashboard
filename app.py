import streamlit as st
import pandas as pd
import numpy as np
import json
import zipfile
import io
from collections import defaultdict

st.set_page_config(page_title="Maggie Robot Line — Dashboard Generator", page_icon="🏭", layout="centered")

# ── Lookups ────────────────────────────────────────────────────────────────
DEPAL_MAP = {
    0:'Stopped',1:'RunningAuto',2:'RunningManual',3:'RunningHoming',
    4:'WaitingForPalletiserBaskets',5:'WaitingForDownstream',6:'WaitingForPalletLoaded',
    7:'StoppedNoErrorCode',8:'AxisX_AxisError',9:'AxisX_FBError',10:'AxisX_ConfigError',
    11:'AxisY_AxisError',12:'AxisY_FBError',13:'AxisY_ConfigError',14:'AxisZ_AxisError',
    15:'AxisZ_FBError',16:'AxisZ_ConfigError',17:'AxisRot_AxisError',18:'AxisRot_FBError',
    19:'AxisRot_ConfigError',20:'EngineerCalibReq',21:'DrivesEnableException',
    22:'DrivesHomingException',23:'ATV_ProductHozAlarm',24:'InvalidPalletSelection',
    25:'LHSPartialPalletRemoved',26:'RHSPartialPalletRemoved',27:'LHSPalletSensorFault',
    28:'RHSPalletSensorFault',29:'LHSPalletNotRemoved',30:'RHSPalletNotRemoved',
    31:'LHSPalletNotLoaded',32:'RHSPalletNotLoaded',33:'DryCycleActiveWithPalletDetected',
    34:'RobotMoveFailedInPosition',35:'NoCameraFrameChange',36:'BasketIndivTolerenceError',
    37:'BasketIndivNotLocatedFoundBlob',38:'IncorrectPalletOrientation',
    39:'UnpickableBasketDetected',40:'BasketDetectedNotFlat',41:'RobotFailedToPick',
    42:'RobotFailedToPlace',43:'RobotPickFailsRotCalibration',
    44:'RobotWaitingBasketConveyorClear',45:'RobotWaitingPlacePosClear',
    46:'RobotWaitingProductDemandFeedrate',47:'GripperBaleArmRetractLHSTimeout',
    48:'GripperBaleArmRetractRHSTimeout',49:'GripperBaleArmExtendLHSTimeout',
    50:'GripperBaleArmExtendRHSTimeout',51:'GripperTipTimeout',52:'GripperUnTipTimeout',
    53:'CameraNotRunning',54:'CameraModbusError',55:'EncoderCalib_PowerCycleReq',
    56:'CollisionPossibleOverPickX',57:'CollisionPossibleOverPickY',
    58:'CollisionPossibleChangingCells',59:'Basket2DetectionDiscrepancy',
    60:'BasketRotatedTooMuch',61:'BasketExpectedNotSeen',62:'EmptyLayerScan',
}
MACHINE_MAP  = {0:'NotRequired',1:'InFault',2:'Stopped',3:'Warning',4:'Ready'}
ZONE_MAP     = {0:'Disabled',1:'InFault',2:'Blocked',3:'HeldByMachine',4:'Starting',5:'Running',6:'Stopped'}
ERROR_MAP    = {0:'NoError',100:'CheckWeigh',110:'MarkemPrinter',120:'CaseClose',130:'XRay',
                140:'ShrinkWrap',150:'MarkemLabel',160:'DS_Conveyor',170:'EOL_Pall',180:'Carton_Erect'}
SORT_PAL_MAP = {0:'Stopped',1:'RunningAuto',2:'RunningManual',3:'RunningHoming',4:'WaitingForUpstream',5:'WaitingForDownstream'}
ROBOT_FAULT  = {1:'Robot in fault',2:'Curve module fault',3:'Guard door opened'}
OPCODES      = {0,1,2,3,4,5,6,7}
CR_COLS      = ['F12_Clearout','F34_Clearout','F12_Runout','F34_Runout']
DEPAL_COLS   = ['F1_Depal_Status','F2_Depal_Status','F3_Depal_Status','F4_Depal_Status']

def decode(val, lkp):
    if pd.isna(val): return None
    try: return lkp.get(int(val), f'Code_{int(val)}')
    except: return str(val)

def decode_rl(val):
    if pd.isna(val) or val == 0: return None
    try:
        c = int(val)
        return f"Fl{c//10000} R{(c%10000)//100}: {ROBOT_FAULT.get(c%100, f'Fault {c%100}')}"
    except: return None

def overlaps(s1, e1, s2, e2):
    if e1 is None: e1 = s1 + pd.Timedelta('999h')
    if e2 is None: e2 = s2 + pd.Timedelta('999h')
    return s1 < e2 and s2 < e1

# ── Core analysis ──────────────────────────────────────────────────────────
def analyse_shift(df_raw):
    """Run full analysis on a shift dataframe. Returns dict for JS embedding."""
    if len(df_raw) < 20:
        return None

    df = df_raw.copy()
    state_cols = [c for c in df.columns if c not in ['date (UTC)', 'timestamp']]
    df[state_cols] = df[state_cols].ffill()
    TOTAL_S = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds()
    if TOTAL_S < 60:
        return None

    # Planned periods
    def build_planned(df):
        periods = []; in_p = False; start = None; trig = None
        for _, row in df.iterrows():
            any_cr = any(row.get(c) == True for c in CR_COLS if c in row)
            f12 = row.get('F12_Pick_Belt_Running')
            f34 = row.get('F34_Pick_Belt_Running')
            if not in_p and any_cr and ((f12 == False) or (f34 == False)):
                in_p = True; start = row['timestamp']
                trig = ', '.join(c for c in CR_COLS if c in row and row.get(c) == True)
            elif in_p and f12 == True and f34 == True:
                periods.append({'start': start, 'end': row['timestamp'],
                                 'duration_s': (row['timestamp'] - start).total_seconds(), 'trigger': trig})
                in_p = False
        if in_p:
            periods.append({'start': start, 'end': df['timestamp'].iloc[-1],
                             'duration_s': (df['timestamp'].iloc[-1] - start).total_seconds(), 'trigger': trig})
        return periods

    planned = build_planned(df)

    def in_pl(ts):
        return any(p['start'] <= ts <= p['end'] for p in planned)

    def is_pl(ts):
        for p in planned:
            if p['start'] <= ts <= p['end']: return True, p['trigger']
        return False, None

    # Build fault events
    def build_events(sub, is_f, lbl_f):
        evts = []; in_f = False; s_ts = None; s_lbl = None
        for _, row in sub.iterrows():
            fault = is_f(row) and not in_pl(row['timestamp'])
            lbl = lbl_f(row) if is_f(row) else None
            if fault and not in_f:
                in_f = True; s_ts = row['timestamp']; s_lbl = lbl
            elif not fault and in_f:
                evts.append({'start': s_ts, 'end': row['timestamp'],
                              'duration_s': (row['timestamp'] - s_ts).total_seconds(), 'label': s_lbl})
                in_f = False
            elif fault and in_f and lbl != s_lbl:
                evts.append({'start': s_ts, 'end': row['timestamp'],
                              'duration_s': (row['timestamp'] - s_ts).total_seconds(), 'label': s_lbl})
                s_ts = row['timestamp']; s_lbl = lbl
        if in_f:
            evts.append({'start': s_ts, 'end': df['timestamp'].iloc[-1],
                          'duration_s': (df['timestamp'].iloc[-1] - s_ts).total_seconds(), 'label': s_lbl})
        return evts

    def summarise(evts):
        d = defaultdict(lambda: {'count': 0, 'total_s': 0, 'max_s': 0})
        for e in evts:
            lbl = e['label'] or 'Unknown'
            d[lbl]['count'] += 1; d[lbl]['total_s'] += e['duration_s']
            d[lbl]['max_s'] = max(d[lbl]['max_s'], e['duration_s'])
        return {k: {**v, 'avg_s': v['total_s'] / v['count']} for k, v in d.items()}

    mg = {}
    # Depals
    for i, col in enumerate(DEPAL_COLS, 1):
        if col not in df.columns: continue
        sub = df[['timestamp', col]].dropna(subset=[col]).copy()
        evts = build_events(sub,
            lambda r, c=col: pd.notna(r[c]) and int(r[c]) >= 7,
            lambda r, c=col: f"{int(r[c])} – {decode(r[c], DEPAL_MAP)}")
        mg[f'Depal F{i}'] = {'events': evts, 'summary': summarise(evts)}

    # Combo machine+zone
    for name, (mc, zc) in [
        ('Case Closer',    ('Case_Closer_Machine_Status',   'Case_Closer_Zone_2_Status')),
        ('Check Weigher',  ('Check_Weigher_Machine_Status', 'Check_Weigher_Zone_1_Status')),
        ('XRay',           ('XRay_Machine_Status',          'XRay_Zone_3_Status')),
        ('Shrink Wrapper', ('Shrink_Wrapper_Machine_Status','Shrink_Wrapper_Zone_4_Status')),
    ]:
        if mc not in df.columns or zc not in df.columns: continue
        sub = df[['timestamp', mc, zc]].copy()
        evts = build_events(sub,
            lambda r, m=mc, z=zc: (pd.notna(r[m]) and int(r[m]) in [1,2]) or (pd.notna(r[z]) and int(r[z]) in [1,2,6]),
            lambda r, m=mc, z=zc: ' | '.join(filter(None, [
                f"Machine:{decode(r[m], MACHINE_MAP)}" if pd.notna(r[m]) and int(r[m]) in [1,2] else None,
                f"Zone:{decode(r[z], ZONE_MAP)}" if pd.notna(r[z]) and int(r[z]) in [1,2,6] else None
            ])) or 'Unknown')
        mg[name] = {'events': evts, 'summary': summarise(evts)}

    # Single-column machines
    for nm, col, is_zone in [
        ('Markem Labeller',    'Markem_Labeller_Machine_Status', False),
        ('Downstream Palletiser', 'Downstream_Pal_Zone_5_Status', True),
    ]:
        if col not in df.columns: continue
        sub = df[['timestamp', col]].dropna(subset=[col]).copy()
        fault_vals = [1,2,6] if is_zone else [1,2]
        lbl_map = ZONE_MAP if is_zone else MACHINE_MAP
        prefix = 'Zone' if is_zone else 'Machine'
        evts = build_events(sub,
            lambda r, c=col, fv=fault_vals: pd.notna(r[c]) and int(r[c]) in fv,
            lambda r, c=col, lm=lbl_map, p=prefix: f"{p}:{decode(r[c], lm)}")
        mg[nm] = {'events': evts, 'summary': summarise(evts)}

    # Robot lines
    rl_data = {}
    for col in ['F1_RobotLine_Status','F2_RobotLine_Status','F3_RobotLine_Status','F4_RobotLine_Status']:
        if col not in df.columns: continue
        sub = df[['timestamp', col]].dropna(subset=[col]).copy()
        evts = build_events(sub,
            lambda r, c=col: pd.notna(r[c]) and r[c] != 0,
            lambda r, c=col: decode_rl(r[c]) or f"Code_{int(r[c])}")
        rl_data[col] = {'events': evts, 'summary': summarise(evts)}

    # Robot productive states
    rp_data = {}
    for col in [c for c in df.columns if 'ProductiveState' in c]:
        sub = df[['timestamp', col]].dropna(subset=[col]).copy()
        st = {0:0,1:0,2:0,3:0,4:0}; prev_ts = None; prev_val = None
        for _, row in sub.iterrows():
            if prev_val is not None and not in_pl(row['timestamp']):
                dur = (row['timestamp'] - prev_ts).total_seconds()
                st[int(prev_val)] = st.get(int(prev_val), 0) + dur
            prev_ts = row['timestamp']; prev_val = row[col]
        rp_data[col] = {'times': st}

    # Depal raw changes for lookback
    raw_dc = {col: df_raw[df_raw[col].notna()][['timestamp', col]].copy().reset_index(drop=True)
              for col in DEPAL_COLS if col in df_raw.columns}

    def get_depal_lb(col, start_ts):
        if col not in raw_dc: return None, None
        ch = raw_dc[col]
        sv = ch[(ch[col]==7) &
                (ch['timestamp'] >= start_ts - pd.Timedelta('30s')) &
                (ch['timestamp'] <= start_ts + pd.Timedelta('30s'))].copy()
        if len(sv) == 0: return None, None
        sv['dist'] = (sv['timestamp'] - start_ts).abs()
        s_ts = sv.sort_values('dist').iloc[0]['timestamp']
        win = ch[(ch['timestamp'] >= s_ts - pd.Timedelta('30s')) &
                 (ch['timestamp'] <= s_ts + pd.Timedelta('10s')) &
                 (ch['timestamp'] != s_ts)].copy()
        win['dist'] = (win['timestamp'] - s_ts).abs(); win = win.sort_values('dist')
        for _, row in win.iterrows():
            iv = int(row[col])
            if iv not in OPCODES: return iv, decode(row[col], DEPAL_MAP)
        return None, None

    def get_cause(start_ts):
        w = df[(df['timestamp'] >= start_ts - pd.Timedelta('10s')) &
               (df['timestamp'] <= start_ts + pd.Timedelta('10s'))].copy()
        causes = []; evidence = []
        if 'Overall_Error_Code' in w.columns:
            nz = w['Overall_Error_Code'].dropna(); nz = nz[nz != 0]
            if len(nz) > 0:
                desc = decode(nz.iloc[-1], ERROR_MAP)
                causes.append(f"Downstream error: {desc}"); evidence.append(f"OEC={int(nz.iloc[-1])} ({desc})")
        for col in ['Case_Closer_Machine_Status','Check_Weigher_Machine_Status',
                    'Markem_Labeller_Machine_Status','Shrink_Wrapper_Machine_Status','XRay_Machine_Status']:
            if col in w.columns and len(w[col].dropna()[w[col].dropna()==1]) > 0:
                causes.append(f"{col.replace('_Machine_Status','').replace('_',' ')} in fault")
                evidence.append(f"{col}=1")
        for col in ['Check_Weigher_Zone_1_Status','Case_Closer_Zone_2_Status','XRay_Zone_3_Status',
                    'Shrink_Wrapper_Zone_4_Status','Downstream_Pal_Zone_5_Status']:
            if col not in w.columns: continue
            fv = w[col].dropna(); fv = fv[fv.isin([1,2])]
            if len(fv) > 0:
                desc = decode(fv.iloc[-1], ZONE_MAP)
                causes.append(f"{col.replace('_Status','').replace('_',' ')}: {desc}"); evidence.append(f"{col}={int(fv.iloc[-1])}")
        for col in DEPAL_COLS:
            if col not in w.columns: continue
            fv = w[col].dropna(); fv = fv[fv >= 7]
            if len(fv) > 0:
                rc = int(fv.iloc[-1])
                if rc == 7:
                    lb, ld = get_depal_lb(col, start_ts)
                    if lb: causes.append(f"{col.replace('_Status','')}: {ld} (cleared→StoppedNoErrorCode)"); evidence.append(f"{col}=7→{lb}")
                    else: causes.append(f"{col.replace('_Status','')}: StoppedNoErrorCode"); evidence.append(f"{col}=7")
                else:
                    desc = decode(rc, DEPAL_MAP)
                    causes.append(f"{col.replace('_Status','')}: {desc}"); evidence.append(f"{col}={rc}")
        for col in ['F1_RobotLine_Status','F2_RobotLine_Status','F3_RobotLine_Status','F4_RobotLine_Status']:
            if col not in w.columns: continue
            fv = w[col].dropna(); fv = fv[fv != 0]
            if len(fv) > 0:
                desc = decode_rl(fv.iloc[-1])
                causes.append(f"{col.replace('_RobotLine_Status','')} Robot fault: {desc}"); evidence.append(f"{col}={int(fv.iloc[-1])}")
        for col in ['F1_Sortation_Status','F2_Sortation_Status','Palletiser_Status','Stacker_Status']:
            if col not in w.columns: continue
            wv = w[col].dropna(); wv = wv[wv.isin([4,5])]
            if len(wv) > 0:
                desc = decode(wv.iloc[-1], SORT_PAL_MAP)
                causes.append(f"{col.replace('_Status','')}: {desc}"); evidence.append(f"{col}={int(wv.iloc[-1])}")
        if 'F34_Carton_Queue_Full' in w.columns and True in w['F34_Carton_Queue_Full'].dropna().values:
            causes.append("F34 Carton Queue Full")
        if 'F34_Carton_Queue_Empty' in w.columns and True in w['F34_Carton_Queue_Empty'].dropna().values:
            causes.append("F34 Carton Queue Empty")
        return (causes[0], '; '.join(evidence)) if causes else ('Unknown', 'No changes in ±10s')

    def build_stops(belt):
        if belt not in df.columns: return []
        stops = []; in_s = False; s_ts = None
        evts = df[df[belt].notna()][['timestamp', belt]].copy()
        for _, row in evts.iterrows():
            if row[belt] == False and not in_s: in_s = True; s_ts = row['timestamp']
            elif row[belt] == True and in_s:
                stops.append({'start': s_ts, 'end': row['timestamp'],
                               'duration_s': (row['timestamp'] - s_ts).total_seconds()})
                in_s = False
        if in_s: stops.append({'start': s_ts, 'end': None, 'duration_s': None})
        return stops

    f12 = build_stops('F12_Pick_Belt_Running')
    f34 = build_stops('F34_Pick_Belt_Running')

    records = []
    for s in f12:
        f34_ov = any(overlaps(s['start'], s['end'], s2['start'], s2['end']) for s2 in f34)
        st = 'Full Stop' if f34_ov else 'Partial Stop (F12 only)'
        pl, trig = is_pl(s['start'])
        if pl: records.append({'Type': st, 'Cause': 'Planned Stoppage', 'Duration_s': s['duration_s']})
        else:
            c, ev = get_cause(s['start'])
            records.append({'Type': st, 'Cause': c, 'Duration_s': s['duration_s']})
    for s in f34:
        if not any(overlaps(s['start'], s['end'], s2['start'], s2['end']) for s2 in f12):
            pl, trig = is_pl(s['start'])
            if pl: records.append({'Type': 'Partial Stop (F34 only)', 'Cause': 'Planned Stoppage', 'Duration_s': s['duration_s']})
            else:
                c, ev = get_cause(s['start'])
                records.append({'Type': 'Partial Stop (F34 only)', 'Cause': c, 'Duration_s': s['duration_s']})

    results = pd.DataFrame(records)
    if len(results) > 0: results['Duration_s'] = results['Duration_s'].round(1)
    plan_df  = results[results['Cause'] == 'Planned Stoppage'] if len(results) > 0 else pd.DataFrame()
    full_df  = results[results['Type'] == 'Full Stop'] if len(results) > 0 else pd.DataFrame()
    uplan_df = full_df[full_df['Cause'] != 'Planned Stoppage'] if len(full_df) > 0 else pd.DataFrame()

    def st_tot(summary):
        if not summary: return 0, 0
        return sum(v['count'] for v in summary.values()), sum(v['total_s'] for v in summary.values())

    MJ = {n: {'total_s': round(st_tot(d['summary'])[1], 1), 'count': st_tot(d['summary'])[0],
               'breakdown': {l: {'c': v['count'], 't': round(v['total_s'], 1)}
                              for l, v in sorted(d['summary'].items(), key=lambda x: -x[1]['total_s'])}}
          for n, d in mg.items()}
    RJ = {col.replace('_RobotLine_Status', '') + ' Line':
          {'total_s': round(st_tot(d['summary'])[1], 1), 'count': st_tot(d['summary'])[0],
           'breakdown': {l: {'c': v['count'], 't': round(v['total_s'], 1)}
                         for l, v in sorted(d['summary'].items(), key=lambda x: -x[1]['total_s'])}}
          for col, d in rl_data.items()}
    PJ = {col.replace('_ProductiveState', '').replace('_', ' '):
          {'A': round(d['times'].get(4,0),1), 'WP': round(d['times'].get(2,0),1),
           'WC': round(d['times'].get(1,0),1), 'NP': round(d['times'].get(0,0),1)}
          for col, d in rp_data.items()}

    return {
        'total_s':        round(TOTAL_S, 1),
        'machines':       MJ,
        'robot_lines':    RJ,
        'robot_prod':     PJ,
        'planned_count':  len(plan_df),
        'planned_s':      round(float(plan_df['Duration_s'].sum()), 1) if len(plan_df) > 0 else 0,
        'unplanned_count': len(uplan_df),
        'unplanned_s':    round(float(uplan_df['Duration_s'].sum()), 1) if len(uplan_df) > 0 else 0,
    }


# ── HTML template ──────────────────────────────────────────────────────────
def build_html(label, badge_class, shift_type, ps, pe, hrs, data):
    TS = data['total_s']
    MJ = json.dumps(data['machines'])
    RJ = json.dumps(data['robot_lines'])
    PJ = json.dumps(data['robot_prod'])
    PC = data['planned_count']
    PS = data['planned_s']
    UC = data['unplanned_count']
    US = data['unplanned_s']

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Maggie — {label}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
  *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0;}}
  body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:#f5f5f3;color:#1a1a18;min-height:100vh;}}
  .wrap{{max-width:960px;margin:0 auto;padding:2rem 1.25rem;}}
  h1{{font-size:20px;font-weight:500;margin-bottom:4px;}}
  .subtitle{{font-size:13px;color:#73726c;margin-bottom:1.5rem;}}
  .shift-badge{{display:inline-block;padding:3px 10px;border-radius:4px;font-size:12px;font-weight:500;margin-bottom:8px;}}
  .shift-day{{background:#EBF3FF;color:#185FA5;}}
  .shift-night{{background:#2C2C2A;color:#D3D1C7;}}
  .tab-bar{{display:flex;gap:4px;flex-wrap:wrap;background:#e8e6df;padding:4px;border-radius:10px;width:fit-content;margin-bottom:1.5rem;}}
  .tab{{padding:6px 16px;font-size:13px;border-radius:8px;cursor:pointer;border:1px solid transparent;background:transparent;color:#73726c;font-family:inherit;transition:all 0.15s;}}
  .tab.active{{background:#fff;border-color:#d3d1c7;color:#1a1a18;font-weight:500;}}
  .tab:hover:not(.active){{background:#dddbd4;}}
  .card{{background:#fff;border:0.5px solid #d3d1c7;border-radius:12px;padding:1.25rem;margin-bottom:1rem;}}
  .metric-grid{{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:1.25rem;}}
  .metric{{background:#f1efe8;border-radius:8px;padding:1rem;flex:1;min-width:130px;}}
  .metric.planned{{background:#e2efda;}}
  .metric-label{{font-size:12px;color:#73726c;margin-bottom:4px;}}
  .metric-val{{font-size:20px;font-weight:500;}}
  .metric-sub{{font-size:11px;color:#73726c;margin-top:3px;}}
  .machine-grid{{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:1.25rem;}}
  .mc{{background:#fff;border:0.5px solid #d3d1c7;border-radius:12px;padding:1rem;flex:1;min-width:150px;cursor:pointer;transition:border-color 0.15s;}}
  .mc:hover{{border-color:#888780;}}
  .mc.selected{{border:2px solid #378ADD;}}
  .mc-name{{font-size:13px;font-weight:500;margin-bottom:6px;}}
  .mc-time{{font-size:22px;font-weight:500;}}
  .mc-sub{{font-size:12px;color:#73726c;margin-top:2px;}}
  .section-title{{font-size:13px;font-weight:500;color:#73726c;margin-bottom:1rem;}}
  table{{width:100%;border-collapse:collapse;}}
  th{{font-size:12px;color:#73726c;font-weight:500;padding-bottom:8px;border-bottom:0.5px solid #d3d1c7;}}
  th:first-child{{text-align:left;}}
  th:not(:first-child){{text-align:right;padding-right:8px;}}
  td{{font-size:13px;padding:7px 0;border-bottom:0.5px solid #f1efe8;}}
  td:first-child{{padding-right:12px;}}
  td:not(:first-child){{text-align:right;padding-right:8px;}}
  tr:nth-child(even) td{{background:#fafaf8;}}
  .legend{{display:flex;flex-wrap:wrap;gap:14px;margin-bottom:1rem;}}
  .legend-item{{display:flex;align-items:center;gap:6px;font-size:12px;color:#73726c;}}
  .ldot{{width:10px;height:10px;border-radius:2px;flex-shrink:0;}}
  .srow{{display:flex;align-items:center;gap:12px;padding:8px 0;border-bottom:0.5px solid #f1efe8;}}
  .rname{{font-size:13px;font-weight:500;min-width:70px;}}
  .sbw{{flex:1;display:flex;height:22px;border-radius:4px;overflow:hidden;gap:1px;}}
  .apct{{font-size:13px;font-weight:500;min-width:52px;text-align:right;}}
</style>
</head>
<body>
<div class="wrap">
  <span class="shift-badge {badge_class}">{shift_type}</span>
  <h1>Maggie Robot Line — {label}</h1>
  <p class="subtitle">{ps} – {pe} &nbsp;·&nbsp; {hrs} hours &nbsp;·&nbsp; Planned stoppages excluded from fault analysis</p>
  <div class="tab-bar" id="tabs"></div>
  <div id="content"></div>
</div>
<script>
const TOTAL_S={TS};
const MACHINES={MJ};
const ROBOT_LINES={RJ};
const ROBOT_PROD={PJ};
const PC={PC},PS={PS},UC={UC},US={US};
const COLS=["#E24B4A","#BA7517","#378ADD","#1D9E75","#D4537E","#534AB7","#639922","#D85A30","#0F6E56","#185FA5"];
let charts=[];
function fT(s){{const h=Math.floor(s/3600),m=Math.floor((s%3600)/60),sc=Math.floor(s%60);if(h>0)return h+"h "+m+"m";if(m>0)return m+"m "+sc+"s";return sc+"s";}}
function pct(s){{return((s/TOTAL_S)*100).toFixed(1)+"%";}}
function mn(s){{return Math.round(s/60);}}
function dc(){{charts.forEach(c=>{{try{{c.destroy();}}catch(e){{}}}}); charts=[];}}
function mH(l,v,sub,pl){{return`<div class="metric${{pl?' planned':''}}"><div class="metric-label">${{l}}</div><div class="metric-val">${{v}}</div>${{sub?`<div class="metric-sub">${{sub}}</div>`:''}}` + "</div>";}}
function mkC(id,items){{
  requestAnimationFrame(()=>{{
    const el=document.getElementById(id);if(!el)return;
    charts.push(new Chart(el,{{type:"bar",data:{{labels:items.map(([k])=>k.length>34?k.slice(0,32)+"…":k),datasets:[{{label:"Fault mins",data:items.map(([,d])=>mn(d.t)),backgroundColor:items.map((_,i)=>COLS[i%COLS.length]),borderRadius:3}}]}},options:{{indexAxis:"y",responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},scales:{{x:{{ticks:{{callback:v=>v+"m",font:{{size:11}}}}}},y:{{ticks:{{font:{{size:11}}}}}}}}}}}}));
  }});
}}
function dCard(title,count,ts,items,id){{
  const rows=items.map(([k,d])=>`<tr><td>${{k}}</td><td>${{d.c}}</td><td>${{fT(d.t)}}</td><td>${{fT(Math.round(d.t/d.c))}}</td></tr>`).join("");
  return`<div class="card"><div style="font-size:14px;font-weight:500;margin-bottom:1rem">${{title}}</div>
    <div class="metric-grid">${{mH("Fault events",count)}}${{mH("Total fault time",fT(ts))}}${{mH("% of shift",pct(ts))}}</div>
    <div style="position:relative;height:${{Math.max(180,items.length*34+60)}}px;margin-bottom:1.25rem"><canvas id="${{id}}" role="img" aria-label="Fault breakdown"></canvas></div>
    <table><thead><tr><th>Fault / state</th><th>Events</th><th>Total</th><th>Avg</th></tr></thead><tbody>${{rows}}</tbody></table></div>`;
}}
function renderOverview(){{
  dc();
  const sm=Object.entries(MACHINES).sort((a,b)=>b[1].total_s-a[1].total_s);
  const rl=Object.entries(ROBOT_LINES).sort((a,b)=>b[1].total_s-a[1].total_s);
  const te=Object.values(MACHINES).reduce((a,v)=>a+v.count,0);
  const tf=Object.values(MACHINES).reduce((a,v)=>a+v.total_s,0);
  document.getElementById("content").innerHTML=
    `<div class="metric-grid">
      ${{mH("Machine fault events",te.toLocaleString())}}
      ${{mH("Machine fault time",fT(tf))}}
      ${{mH("Unplanned full stops",UC.toLocaleString(),"both belts stopped")}}
      ${{mH("Unplanned stop time",fT(US))}}
      ${{mH("Planned stoppages",PC.toLocaleString(),"clearout / runout active",true)}}
      ${{mH("Planned stop time",fT(PS),"",true)}}
    </div>
    <div class="card"><div class="section-title">Machine fault time (minutes) — planned periods excluded</div>
      <div style="position:relative;height:${{sm.length*38+60}}px"><canvas id="ov1" role="img" aria-label="Machine faults"></canvas></div></div>
    <div class="card"><div class="section-title">Robot line fault time (minutes) — planned periods excluded</div>
      <div style="position:relative;height:${{rl.length*38+60}}px"><canvas id="ov2" role="img" aria-label="Robot line faults"></canvas></div></div>`;
  requestAnimationFrame(()=>{{
    charts.push(new Chart(document.getElementById("ov1"),{{type:"bar",data:{{labels:sm.map(([k])=>k),datasets:[{{label:"Fault mins",data:sm.map(([,v])=>mn(v.total_s)),backgroundColor:sm.map((_,i)=>COLS[i%COLS.length]),borderRadius:3}}]}},options:{{indexAxis:"y",responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{label:c=>`${{c.raw}} min · ${{sm[c.dataIndex][1].count}} events`}}}}}},scales:{{x:{{ticks:{{callback:v=>v+"m",font:{{size:11}}}}}},y:{{ticks:{{font:{{size:12}}}}}}}}}}}}));
    charts.push(new Chart(document.getElementById("ov2"),{{type:"bar",data:{{labels:rl.map(([k])=>k),datasets:[{{label:"Fault mins",data:rl.map(([,v])=>mn(v.total_s)),backgroundColor:["#E24B4A","#BA7517","#E24B4A","#639922"],borderRadius:3}}]}},options:{{indexAxis:"y",responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{label:c=>`${{c.raw}} min · ${{rl[c.dataIndex][1].count}} events`}}}}}},scales:{{x:{{ticks:{{callback:v=>v+"m",font:{{size:11}}}}}},y:{{ticks:{{font:{{size:12}}}}}}}}}}}}));
  }});
}}
function renderMachines(group){{
  dc();
  const entries=Object.entries(MACHINES).filter(([k])=>group==="depals"?k.startsWith("Depal"):!k.startsWith("Depal"));
  let html=`<div class="machine-grid">`;
  entries.forEach(([name,v])=>{{
    const col=v.total_s>7200?"#E24B4A":v.total_s>2400?"#BA7517":"#1a1a18";
    html+=`<div class="mc" id="mc-${{name.replace(/[\\s+]/g,'_')}}" onclick="showD('${{name}}')"><div class="mc-name">${{name}}</div><div class="mc-time" style="color:${{col}}">${{fT(v.total_s)}}</div><div class="mc-sub">${{v.count}} events · ${{pct(v.total_s)}}</div></div>`;
  }});
  document.getElementById("content").innerHTML=html+`</div><div id="det"></div>`;
}}
function showD(name){{
  dc();
  document.querySelectorAll(".mc").forEach(el=>el.classList.remove("selected"));
  const el=document.getElementById("mc-"+name.replace(/[\\s+]/g,'_'));if(el)el.classList.add("selected");
  const v=MACHINES[name];const items=Object.entries(v.breakdown).sort((a,b)=>b[1].t-a[1].t);
  document.getElementById("det").innerHTML=dCard(name+" — fault breakdown",v.count,v.total_s,items,"dc");
  mkC("dc",items);
}}
function renderRobotLines(){{
  dc();
  let html=`<div class="machine-grid">`;
  Object.entries(ROBOT_LINES).forEach(([name,v])=>{{
    const col=v.total_s>50000?"#E24B4A":v.total_s>10000?"#BA7517":"#1a1a18";
    html+=`<div class="mc" id="rl-${{name.replace(/\\s/g,'_')}}" onclick="showRL('${{name}}')"><div class="mc-name">${{name}}</div><div class="mc-time" style="color:${{col}}">${{fT(v.total_s)}}</div><div class="mc-sub">${{v.count}} events · ${{pct(v.total_s)}}</div></div>`;
  }});
  document.getElementById("content").innerHTML=html+`</div><div id="rld"></div>`;
}}
function showRL(name){{
  dc();
  document.querySelectorAll(".mc").forEach(el=>el.classList.remove("selected"));
  const el=document.getElementById("rl-"+name.replace(/\\s/g,'_'));if(el)el.classList.add("selected");
  const v=ROBOT_LINES[name];const items=Object.entries(v.breakdown).sort((a,b)=>b[1].t-a[1].t);
  document.getElementById("rld").innerHTML=dCard(name+" — fault breakdown",v.count,v.total_s,items,"rlc");
  mkC("rlc",items);
}}
function renderRobotStates(){{
  dc();
  const legend=`<div class="legend"><div class="legend-item"><div class="ldot" style="background:#639922"></div>Active (MotionActive)</div><div class="legend-item"><div class="ldot" style="background:#378ADD"></div>Waiting for product</div><div class="legend-item"><div class="ldot" style="background:#BA7517"></div>Waiting for carton</div><div class="legend-item"><div class="ldot" style="background:#E24B4A"></div>Not productive</div></div>`;
  const rows=Object.entries(ROBOT_PROD).map(([name,d])=>{{
    const total=d.A+d.WP+d.WC+d.NP;
    const ap=(d.A/total*100).toFixed(1);
    const col=ap>70?"#3B6D11":ap>50?"#BA7517":"#E24B4A";
    const segs=[{{w:d.A/total*100,c:"#639922",l:"Active"}},{{w:d.WP/total*100,c:"#378ADD",l:"Wait product"}},{{w:d.WC/total*100,c:"#BA7517",l:"Wait carton"}},{{w:d.NP/total*100,c:"#E24B4A",l:"Not productive"}}];
    const bar=segs.filter(s=>s.w>0.3).map(s=>`<div title="${{s.l}}: ${{s.w.toFixed(1)}}%" style="width:${{s.w}}%;background:${{s.c}};min-width:2px"></div>`).join("");
    return`<div class="srow"><div class="rname">${{name}}</div><div class="sbw">${{bar}}</div><div class="apct" style="color:${{col}}">${{ap}}%</div></div>`;
  }}).join("");
  document.getElementById("content").innerHTML=`<div class="card">${{legend}}<div>${{rows}}</div></div>`;
}}
const TABS=[["overview","Overview"],["depals","Depals"],["downstream","Downstream"],["robotlines","Robot lines"],["robotstates","Robot states"]];
function setTab(t){{
  document.getElementById("tabs").innerHTML=TABS.map(([k,l])=>`<button class="tab ${{t===k?"active":""}}" onclick="setTab('${{k}}')">${{l}}</button>`).join("");
  if(t==="overview")renderOverview();
  else if(t==="depals")renderMachines("depals");
  else if(t==="downstream")renderMachines("downstream");
  else if(t==="robotlines")renderRobotLines();
  else if(t==="robotstates")renderRobotStates();
}}
window.showD=showD;window.showRL=showRL;window.setTab=setTab;
setTab("overview");
</script>
</body>
</html>"""


# ── Streamlit UI ───────────────────────────────────────────────────────────
st.title("🏭 Maggie Robot Line — Dashboard Generator")
st.markdown("Upload a CSV export from the robot line to generate interactive HTML dashboards.")

uploaded = st.file_uploader("Upload CSV file", type=["csv"], help="Standard robot line data export with timestamp column 'date (UTC)'")

if uploaded:
    st.info("Loading file…")
    try:
        df_all = pd.read_csv(uploaded, low_memory=False)
        df_all['timestamp'] = pd.to_datetime(df_all['date (UTC)'])
        df_all = df_all.sort_values('timestamp').reset_index(drop=True)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    t_start = df_all['timestamp'].iloc[0]
    t_end   = df_all['timestamp'].iloc[-1]
    duration_hrs = round((t_end - t_start).total_seconds() / 3600, 1)

    st.success(f"Loaded {len(df_all):,} rows · {t_start.strftime('%d %b %Y %H:%M')} → {t_end.strftime('%d %b %Y %H:%M')} ({duration_hrs} hrs)")

    # Mode selection
    mode = st.radio("Dashboard mode", ["Single dashboard (full period)", "Per shift (Day 07:00–19:00 / Night 19:00–07:00)"], index=0)

    if st.button("Generate dashboards", type="primary"):

        if mode.startswith("Single"):
            with st.spinner("Analysing…"):
                data = analyse_shift(df_all)
            if data is None:
                st.error("Not enough data to generate a dashboard.")
            else:
                ps  = t_start.strftime('%d %b %Y %H:%M')
                pe  = t_end.strftime('%d %b %Y %H:%M')
                hrs = round(data['total_s'] / 3600, 1)
                label = f"{t_start.strftime('%d %b %Y')} — {t_end.strftime('%d %b %Y')}"
                html = build_html(label, "shift-day", "Full period", ps, pe, hrs, data)
                fname = f"maggie_{t_start.strftime('%Y%m%d')}_full.html"
                st.download_button(f"⬇  Download {fname}", data=html.encode(), file_name=fname, mime="text/html")
                st.success("Dashboard ready!")

        else:
            # Per-shift mode
            shifts = []
            d = t_start.normalize()
            while d <= t_end + pd.Timedelta('1D'):
                for h_start, h_end, stype in [(7, 19, 'Day'), (19, 31, 'Night')]:
                    s_start = d + pd.Timedelta(f'{h_start}h')
                    s_end   = d + pd.Timedelta(f'{h_end}h')
                    fname   = f"maggie_{stype.lower()}_{d.strftime('%Y%m%d')}.html"
                    label   = f"{stype} shift — {d.strftime('%a %d %b %Y')}"
                    shifts.append((s_start, s_end, stype, label, fname))
                d += pd.Timedelta('1D')

            results = []
            progress = st.progress(0)
            status   = st.empty()
            valid = [(s,e,st_,l,f) for s,e,st_,l,f in shifts
                     if len(df_all[(df_all['timestamp']>=s)&(df_all['timestamp']<e)]) >= 50]

            for i, (s_start, s_end, stype, label, fname) in enumerate(valid):
                status.text(f"Processing {label}…")
                mask  = (df_all['timestamp'] >= s_start) & (df_all['timestamp'] < s_end)
                df_sh = df_all[mask].copy().reset_index(drop=True)
                data  = analyse_shift(df_sh)
                if data:
                    ps  = df_sh['timestamp'].iloc[0].strftime('%d %b %Y %H:%M')
                    pe  = df_sh['timestamp'].iloc[-1].strftime('%d %b %Y %H:%M')
                    hrs = round(data['total_s'] / 3600, 1)
                    badge = "shift-day" if stype == "Day" else "shift-night"
                    badge_txt = "Day shift  07:00–19:00" if stype == "Day" else "Night shift  19:00–07:00"
                    html = build_html(label, badge, badge_txt, ps, pe, hrs, data)
                    results.append((fname, html, label, data['unplanned_count']))
                progress.progress((i + 1) / len(valid))

            status.empty(); progress.empty()

            if results:
                # Bundle into a zip
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for fname, html, _, _ in results:
                        zf.writestr(fname, html)
                zip_buf.seek(0)

                zip_name = f"maggie_shifts_{t_start.strftime('%Y%m%d')}.zip"
                st.download_button(f"⬇  Download all {len(results)} shift dashboards (.zip)",
                                   data=zip_buf, file_name=zip_name, mime="application/zip")

                st.success(f"Generated {len(results)} shift dashboards")
                st.markdown("**Shifts included:**")
                for fname, _, label, uc in results:
                    icon = "🌤" if "Day" in label else "🌙"
                    st.markdown(f"- {icon} **{label}** — {uc} unplanned full stops")
            else:
                st.error("No valid shifts found in the data.")
