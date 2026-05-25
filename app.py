from flask import Flask, render_template, request
import numpy as np
import skfuzzy as fuzz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64, math, csv
import pandas as pd
from flask import Response

app = Flask(__name__)

# ============================================================
#  Konstanta & Parameter MF
# ============================================================
RISK_UNIVERSE = np.arange(0, 101, 0.1)
AGE_UNIVERSE = np.arange(10, 121, 1)
HEIGHT_UNIVERSE = np.arange(100, 221, 1)
WEIGHT_UNIVERSE = np.arange(5, 131, 1)

AGE_MF = {
    'muda':   ('trap', [10, 10, 20, 35]),
    'dewasa': ('trap', [25, 35, 50, 60]),
    'tua':    ('trap', [50, 60, 120, 120]),
}

IMT_MF = {
    'kurang': ('trap', [0, 0, 18.49, 18.51]),
    'normal': ('trap', [18.49, 18.51, 22.99, 23.01]),
    'risiko': ('trap', [22.99, 23.01, 24.99, 25.01]),
    'obese1': ('trap', [24.99, 25.01, 29.99, 30.01]),
    'obese2': ('trap', [29.99, 30.01, 100, 100]),
}

RISK_MF = {
    'kurang':   ('trap', [0,  0,  15, 25]),
    'normal':   ('tri',  [20, 36, 47]),
    'berlebih': ('tri',  [43, 58, 70]),
    'obese1':   ('tri',  [65, 79, 88]),
    'obese2':   ('trap', [87, 93, 100, 100]),
}

RULES_MALE = [
    # IMT Kurang
    ('muda', 'kurang', 'kurang'),
    ('dewasa', 'kurang', 'kurang'),
    ('tua', 'kurang', 'kurang'),
    # IMT Normal
    ('muda', 'normal', 'normal'),
    ('dewasa', 'normal', 'normal'),
    ('tua', 'normal', 'normal'),
    # IMT Dengan Risiko
    ('muda', 'risiko', 'berlebih'),
    ('dewasa', 'risiko', 'berlebih'),
    ('tua', 'risiko', 'berlebih'),
    # IMT Obesitas I
    ('muda', 'obese1', 'obese1'),
    ('dewasa', 'obese1', 'obese1'),
    ('tua', 'obese1', 'obese1'),
    # IMT Obesitas II
    ('muda', 'obese2', 'obese2'),
    ('dewasa', 'obese2', 'obese2'),
    ('tua', 'obese2', 'obese2'),
]

RULES_FEMALE = RULES_MALE.copy()

LABEL = {
    'muda': 'Muda', 'dewasa': 'Dewasa', 'tua': 'Tua',
    'kurang': 'Underweight', 'normal': 'Normal', 'risiko': 'Dengan Risiko',
    'berlebih': 'Dengan Risiko', 'obese1': 'Obesitas I', 'obese2': 'Obesitas II',
}

TABEL_DIAGNOSIS = {
    "Underweight": ("Underweight", "Risiko malnutrisi. Tingkatkan asupan kalori bernutrisi.", "sedang"),
    "Normal": ("Normal", "Pertahankan pola hidup sehat dan aktif.", "rendah"),
    "Dengan Risiko": ("Dengan Risiko", "Modifikasi gaya hidup: diet gizi seimbang + olahraga.", "sedang"),
    "Obesitas I": ("Obesitas I", "Risiko tinggi. Lakukan diet ketat dan olahraga teratur.", "tinggi"),
    "Obesitas II": ("Obesitas II", "Risiko sangat tinggi. Evaluasi klinis dan pertimbangkan terapi spesifik.", "sangat_tinggi"),
}


# ============================================================
#  Helper: hitung μ
# ============================================================
def mu(val, mf_type, params):
    x = np.array([val])
    if mf_type == 'tri':
        return float(fuzz.trimf(x, params)[0])
    return float(fuzz.trapmf(x, params)[0])


def mu_array(z, mf_type, params):
    if mf_type == 'tri':
        return fuzz.trimf(z, params)
    return fuzz.trapmf(z, params)


# ============================================================
#  STEP 0: Fungsi Keanggotaan — rumus piecewise
# ============================================================
def get_mf_formulas(name, mf_type, params):
    """Return list of {'cond': ..., 'expr': ...} for piecewise formula."""
    rows = []
    if mf_type == 'trap':
        a, b, c, d = params
        if a == b:
            rows.append(dict(cond=f'x ≤ {c}', expr='1'))
        else:
            rows.append(dict(cond=f'x ≤ {a}', expr='0'))
            rows.append(dict(cond=f'{a} < x < {b}', expr=f'(x − {a}) / ({b} − {a})'))
            rows.append(dict(cond=f'{b} ≤ x ≤ {c}', expr='1'))
        if c < d:
            rows.append(dict(cond=f'{c} < x < {d}', expr=f'({d} − x) / ({d} − {c})'))
            rows.append(dict(cond=f'x ≥ {d}', expr='0'))
        elif c == d and a != b:
            pass  # bahu kanan, sudah 1
    elif mf_type == 'tri':
        a, b, c = params
        rows.append(dict(cond=f'x ≤ {a}', expr='0'))
        rows.append(dict(cond=f'{a} < x < {b}', expr=f'(x − {a}) / ({b} − {a})'))
        rows.append(dict(cond=f'{b} ≤ x < {c}', expr=f'({c} − x) / ({c} − {b})'))
        rows.append(dict(cond=f'x ≥ {c}', expr='0'))
    return rows


# ============================================================
#  STEP 1: Fuzzifikasi — substitusi rumus
# ============================================================
def fuzz_detail(val, name, mf_type, params):
    """Return dict with 'mu', 'calc' (substitution text)."""
    v = round(val, 2)
    result = round(mu(val, mf_type, params), 4)

    if mf_type == 'trap':
        a, b, c, d = params
        if a == b:
            if val <= c:
                calc = f'μ({v}) = 1 (karena {v} ≤ {c})'
            elif c < d and val < d:
                num = round(d - val, 4); den = round(d - c, 4)
                calc = f'μ({v}) = ({d} − {v}) / ({d} − {c}) = {num}/{den} = {result}'
            else:
                calc = f'μ({v}) = 0 (karena {v} ≥ {d})'
        elif val <= a:
            calc = f'μ({v}) = 0 (karena {v} ≤ {a})'
        elif val < b:
            num = round(val - a, 4); den = round(b - a, 4)
            calc = f'μ({v}) = ({v} − {a}) / ({b} − {a}) = {num}/{den} = {result}'
        elif val <= c:
            calc = f'μ({v}) = 1 (karena {b} ≤ {v} ≤ {c})'
        elif c == d:
            calc = f'μ({v}) = 1 (karena {v} ≥ {c})'
        elif val < d:
            num = round(d - val, 4); den = round(d - c, 4)
            calc = f'μ({v}) = ({d} − {v}) / ({d} − {c}) = {num}/{den} = {result}'
        else:
            calc = f'μ({v}) = 0 (karena {v} ≥ {d})'
    else:  # tri
        a, b, c = params
        if val <= a:
            calc = f'μ({v}) = 0 (karena {v} ≤ {a})'
        elif val < b:
            num = round(val - a, 4); den = round(b - a, 4)
            calc = f'μ({v}) = ({v} − {a}) / ({b} − {a}) = {num}/{den} = {result}'
        elif val == b:
            calc = f'μ({v}) = 1 (puncak)'
        elif val < c:
            num = round(c - val, 4); den = round(c - b, 4)
            calc = f'μ({v}) = ({c} − {v}) / ({c} − {b}) = {num}/{den} = {result}'
        else:
            calc = f'μ({v}) = 0 (karena {v} ≥ {c})'

    return dict(mu=result, calc=calc)


# ============================================================
#  STEP 3: Titik Potong (clip points)
# ============================================================
def calc_clip_points(risk_key, alpha):
    """Hitung titik potong (clip point) untuk MF output yang di-clip pada alpha."""
    mf_type, params = RISK_MF[risk_key]
    points = []

    if alpha <= 0 or alpha >= 1:
        return points

    if mf_type == 'tri':
        a, b, c = params
        # left clip: (t-a)/(b-a) = alpha  =>  t = a + alpha*(b-a)
        t_left = a + alpha * (b - a)
        points.append(dict(
            name='t_kiri',
            value=round(t_left, 2),
            formula=f'(t − {a}) / ({b} − {a}) = {alpha}',
            steps=f't − {a} = ({b} − {a}) × {alpha} = {round((b-a)*alpha, 4)}',
            result=f't = {a} + {round((b-a)*alpha, 4)} = {round(t_left, 2)}'
        ))
        # right clip: (c-t)/(c-b) = alpha  =>  t = c - alpha*(c-b)
        t_right = c - alpha * (c - b)
        points.append(dict(
            name='t_kanan',
            value=round(t_right, 2),
            formula=f'({c} − t) / ({c} − {b}) = {alpha}',
            steps=f'{c} − t = ({c} − {b}) × {alpha} = {round((c-b)*alpha, 4)}',
            result=f't = {c} − {round((c-b)*alpha, 4)} = {round(t_right, 2)}'
        ))
    elif mf_type == 'trap':
        a, b, c, d = params
        if a < b:
            t_left = a + alpha * (b - a)
            points.append(dict(
                name='t_kiri',
                value=round(t_left, 2),
                formula=f'(t − {a}) / ({b} − {a}) = {alpha}',
                steps=f't − {a} = ({b} − {a}) × {alpha} = {round((b-a)*alpha, 4)}',
                result=f't = {a} + {round((b-a)*alpha, 4)} = {round(t_left, 2)}'
            ))
        if c < d:
            t_right = d - alpha * (d - c)
            points.append(dict(
                name='t_kanan',
                value=round(t_right, 2),
                formula=f'({d} − t) / ({d} − {c}) = {alpha}',
                steps=f'{d} − t = ({d} − {c}) × {alpha} = {round((d-c)*alpha, 4)}',
                result=f't = {d} − {round((d-c)*alpha, 4)} = {round(t_right, 2)}'
            ))

    return points


# ============================================================
#  STEP 4: Segmented Defuzzification
# ============================================================
def build_aggregated(agregasi):
    """Build the aggregated membership function (MAX of all clipped MFs)."""
    z = RISK_UNIVERSE.copy()
    agg = np.zeros_like(z)
    for risk_key, alpha in agregasi.items():
        if alpha <= 0:
            continue
        mf_type, params = RISK_MF[risk_key]
        full = mu_array(z, mf_type, params)
        clipped = np.fmin(alpha, full)
        agg = np.fmax(agg, clipped)
    return z, agg


def find_segments(z, agg, agregasi):
    """Find breakpoints and compute M, A for each segment."""
    # Collect all breakpoints from active clipped MFs
    bps = set()
    for risk_key, alpha in agregasi.items():
        if alpha <= 0:
            continue
        mf_type, params = RISK_MF[risk_key]
        if mf_type == 'tri':
            a, b, c = params
            bps.update([a, c])
            if 0 < alpha < 1:
                bps.add(round(a + alpha * (b - a), 4))
                bps.add(round(c - alpha * (c - b), 4))
            else:
                bps.add(b)
        else:
            a, b, c, d = params
            bps.update([a, d])
            if a < b:
                bps.add(round(a + alpha * (b - a), 4) if 0 < alpha < 1 else b)
            if c < d:
                bps.add(round(d - alpha * (d - c), 4) if 0 < alpha < 1 else c)
            bps.update([b, c])

    bps = sorted([bp for bp in bps if 0 <= bp <= 100])

    # Only keep breakpoints where agg > 0 nearby
    active_bps = []
    for bp in bps:
        idx = np.argmin(np.abs(z - bp))
        lo = max(0, idx - 5)
        hi = min(len(z), idx + 5)
        if np.max(agg[lo:hi]) > 0.0001:
            active_bps.append(bp)

    if len(active_bps) < 2:
        return []

    segments = []
    for i in range(len(active_bps) - 1):
        z_lo = active_bps[i]
        z_hi = active_bps[i + 1]
        if z_hi <= z_lo:
            continue
        mask = (z >= z_lo - 0.01) & (z <= z_hi + 0.01)
        z_seg = z[mask]
        mu_seg = agg[mask]
        if len(z_seg) < 2 or np.max(mu_seg) < 0.0001:
            continue
        mi = float(np.trapezoid(mu_seg * z_seg, z_seg))
        ai = float(np.trapezoid(mu_seg, z_seg))
        segments.append(dict(
            z_start=round(z_lo, 2),
            z_end=round(z_hi, 2),
            momen=round(mi, 4),
            luas=round(ai, 4),
        ))

    return segments


# ============================================================
#  Chart generators
# ============================================================
BG = '#131327'
FACE = '#0e0e20'
GRID = '#252545'
COLORS = ['#00d4aa', '#4fc3f7', '#ffa726', '#ff7043', '#ef5350']


def _style_ax(ax):
    ax.tick_params(colors='#8888aa', labelsize=8)
    ax.set_ylim(-0.05, 1.15)
    ax.grid(True, alpha=0.15, color=GRID)
    for s in ax.spines.values():
        s.set_color('#2a2a4a')
    ax.xaxis.label.set_color('#8888aa')
    ax.yaxis.label.set_color('#8888aa')


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def chart_mf_generic(val, title, xlabel, mf_dict, universe, line_x=True):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    fig.patch.set_facecolor(BG); ax.set_facecolor(FACE)
    for idx, (name, (mt, p)) in enumerate(mf_dict.items()):
        y = mu_array(universe, mt, p)
        ax.plot(universe, y, lw=2, color=COLORS[idx % len(COLORS)], label=LABEL.get(name, name))
        ax.fill_between(universe, y, alpha=0.08, color=COLORS[idx % len(COLORS)])
    if line_x:
        ax.axvline(x=val, color='#fff', ls='--', lw=1.5, alpha=.8, label=f'Input={val:.1f}')
    ax.set_title(title, color='#e0e0e0', fontsize=11, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel('μ(x)', fontsize=10, color='#8888aa')
    _style_ax(ax)
    ax.legend(fontsize=7, loc='upper right', facecolor='#1a1a35', edgecolor='#2a2a4a', labelcolor='#c0c0d0')
    plt.tight_layout()
    return _fig_to_b64(fig)


def chart_per_rule(rules_data):
    """Generate a grid of per-rule clipping charts."""
    n = len(rules_data)
    cols = 5
    rows = math.ceil(n / cols)
    if rows == 0: rows = 1
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3.2 * rows))
    fig.patch.set_facecolor(BG)
    
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for idx, r in enumerate(rules_data):
        ax = axes_flat[idx]
        ax.set_facecolor(FACE)
        rk = r['risk_key']
        mt, p = RISK_MF[rk]
        y_full = mu_array(RISK_UNIVERSE, mt, p)

        if r['active']:
            clipped = np.fmin(r['alpha'], y_full)
            ax.plot(RISK_UNIVERSE, y_full, lw=1, color='#4fc3f7', alpha=0.4)
            ax.fill_between(RISK_UNIVERSE, clipped, alpha=0.25, color='#00d4aa')
            ax.plot(RISK_UNIVERSE, clipped, lw=2, color='#00d4aa')
            ax.axhline(y=r['alpha'], color='#ffa726', ls='--', lw=1)
            ax.set_title(f"R{r['no']} ✓  α={r['alpha']}", color='#00d4aa', fontsize=8, fontweight='bold')
        else:
            ax.plot(RISK_UNIVERSE, y_full, lw=1, color='#555577')
            ax.set_title(f"R{r['no']} ✗  α=0", color='#ef5350', fontsize=8)

        _style_ax(ax)
        ax.set_xlabel(''); ax.set_ylabel('')
        ax.tick_params(labelsize=6)

    # Hide unused axes
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].axis('off')

    plt.tight_layout(pad=1.0)
    return _fig_to_b64(fig)


def chart_agregasi(z, agg, segments_data, crisp):
    """Aggregated function with breakpoints and centroid."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(FACE)

    ax.fill_between(z, agg, alpha=0.3, color='#00d4aa')
    ax.plot(z, agg, lw=2, color='#00d4aa', label='Agregasi (MAX)')

    # Mark segment boundaries
    for i, seg in enumerate(segments_data):
        for pt in [seg['z_start'], seg['z_end']]:
            idx_pt = np.argmin(np.abs(z - pt))
            mu_pt = agg[idx_pt]
            if mu_pt > 0.001:
                ax.plot(pt, mu_pt, 'o', color='#ffa726', ms=5, zorder=5)
                ax.annotate(f'{pt}', (pt, mu_pt), textcoords='offset points',
                           xytext=(0, 10), fontsize=7, color='#ffa726', ha='center')

    ax.axvline(x=crisp, color='#ff7043', ls='--', lw=2, label=f'Centroid Z*={crisp}')
    ax.set_title('Agregasi & Defuzzifikasi (Centroid)', color='#e0e0e0', fontsize=11, fontweight='bold')
    ax.set_xlabel('z (Risiko)', fontsize=9); ax.set_ylabel('μ(z)', fontsize=10, color='#8888aa')
    _style_ax(ax)
    ax.legend(fontsize=8, loc='upper right', facecolor='#1a1a35', edgecolor='#2a2a4a', labelcolor='#c0c0d0')
    plt.tight_layout()
    return _fig_to_b64(fig)


# ============================================================
#  Main calculation
# ============================================================
def perhitungan_lengkap(age_val, jk, tb_val, bb_val):
    steps = {}
    
    # Hitung IMT
    imt_val = round(bb_val / ((tb_val / 100) ** 2), 2)

    # --- STEP 0: Fungsi Keanggotaan formulas ---
    mf_formulas = {}
    mf_formulas['age'] = {name: get_mf_formulas(name, mt, p) for name, (mt, p) in AGE_MF.items()}
    mf_formulas['imt'] = {name: get_mf_formulas(name, mt, p) for name, (mt, p) in IMT_MF.items()}
    mf_formulas['risiko'] = {name: get_mf_formulas(name, mt, p) for name, (mt, p) in RISK_MF.items()}
    steps['mf_formulas'] = mf_formulas

    # --- STEP 1: Fuzzifikasi ---
    fuzz_age = {name: fuzz_detail(age_val, name, mt, p) for name, (mt, p) in AGE_MF.items()}
    fuzz_imt = {name: fuzz_detail(imt_val, name, mt, p) for name, (mt, p) in IMT_MF.items()}
    steps['fuzzifikasi'] = {'age': fuzz_age, 'imt': fuzz_imt}

    # --- STEP 2: Operasi Logika & Implikasi ---
    rules_data = []
    active_rules = RULES_MALE if jk == 'Male' or jk == 'Laki-laki' else RULES_FEMALE
    for i, (ak, ik, rk) in enumerate(active_rules, 1):
        mu_age = fuzz_age[ak]['mu']
        mu_imt = fuzz_imt[ik]['mu']
        alpha = min(mu_age, mu_imt)
        
        rules_data.append(dict(
            no=i,
            age_key=ak, imt_key=ik, risk_key=rk,
            age_label=LABEL[ak], imt_label=LABEL.get(ik, ik), risk_label=LABEL[rk],
            mu_age=mu_age, mu_imt=mu_imt,
            alpha=round(alpha, 4),
            active=alpha > 0,
            calc=f'α{i} = MIN({mu_age}, {mu_imt}) = {round(alpha, 4)}',
        ))
    steps['rules'] = rules_data

    # --- STEP 3a: Agregasi MAX per output ---
    agregasi = {}
    for r in rules_data:
        k = r['risk_key']
        if k not in agregasi or r['alpha'] > agregasi[k]:
            agregasi[k] = r['alpha']
    agregasi = {k: v for k, v in agregasi.items() if v > 0}
    steps['agregasi'] = agregasi

    # --- STEP 3b: Titik potong ---
    all_clips = {}
    for rk, alpha in agregasi.items():
        clips = calc_clip_points(rk, alpha)
        if clips:
            all_clips[rk] = {'alpha': alpha, 'clips': clips}
    steps['titik_potong'] = all_clips

    # --- STEP 3c: Build aggregated ---
    z, agg = build_aggregated(agregasi)

    # --- STEP 4: Defuzzifikasi segmented ---
    segs = find_segments(z, agg, agregasi)
    total_m = sum(s['momen'] for s in segs)
    total_a = sum(s['luas'] for s in segs)
    crisp = round(total_m / total_a, 2) if total_a > 0 else 0

    steps['segments'] = segs
    steps['defuzz'] = dict(
        total_momen=round(total_m, 4),
        total_luas=round(total_a, 4),
        crisp=crisp,
    )

    # --- Charts ---
    charts = {}
    charts['age']     = chart_mf_generic(age_val, 'Fungsi Keanggotaan Usia', 'Usia (tahun)', AGE_MF, AGE_UNIVERSE)
    import numpy as np
    imt_universe = np.linspace(10, 45, 350)
    charts['imt']     = chart_mf_generic(imt_val, 'Fungsi Keanggotaan IMT', 'IMT', IMT_MF, imt_universe)
    charts['risiko']  = chart_mf_generic(0, 'Fungsi Keanggotaan Risiko (Output)', 'Risiko (0-100)', RISK_MF, RISK_UNIVERSE, line_x=False)
    charts['rules']   = chart_per_rule(rules_data)
    charts['agregasi'] = chart_agregasi(z, agg, segs, crisp)

    return steps, crisp, charts


# ============================================================
#  Routes
# ============================================================
@app.route('/')
def home():
    return render_template('index.html', page='home')

@app.route('/demo')
def demo():
    return render_template('index.html', page='demo', result=None)

@app.route('/upload')
def upload():
    return render_template('index.html', page='upload')

def map_risk_to_label(crisp):
    if crisp < 25.5: return "Underweight"
    elif crisp < 47.0: return "Normal"
    elif crisp < 68.5: return "Dengan Risiko"
    elif crisp < 86.0: return "Obesitas I"
    else:              return "Obesitas II"

@app.route('/process_upload', methods=['POST'])
def process_upload():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    
    if file:
        df = pd.read_csv(file)
        
        outputs = []
        scores = []
        
        for index, row in df.iterrows():
            age_val = float(row.get('Age', 30))
            jk = str(row.get('Gender', 'Male'))
            tb = float(row.get('Height', 170))
            bb = float(row.get('Weight', 70))
            
            steps, crisp, charts = perhitungan_lengkap(age_val, jk, tb, bb)
            
            klas = map_risk_to_label(crisp)
            
            outputs.append(klas)
            scores.append(crisp)
            
        df['Prediksi_Label'] = outputs
        df['Skor_Risiko'] = scores
        
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        return Response(
            csv_buffer.getvalue(),
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename=hasil_prediksi.csv"}
        )

@app.route('/diagnosa', methods=['POST'])
def diagnosa():
    try:
        umur   = float(request.form.get('umur', 30))
        jk     = request.form.get('jenis_kelamin', 'Laki-laki')
        bb     = float(request.form.get('berat_badan', 70))
        tb     = float(request.form.get('tinggi_badan', 170))
    except (ValueError, TypeError):
        return render_template('index.html', page='demo', result=None, error="Input tidak valid.")

    steps, crisp, charts = perhitungan_lengkap(umur, jk, tb, bb)

    klas = map_risk_to_label(crisp)
    diag, saran, level = TABEL_DIAGNOSIS.get(klas, ("Tidak diketahui","Konsultasi dokter.","sedang"))
    
    # Kustomisasi saran berdasarkan Jenis Kelamin (Post-Processing)
    if jk in ['Male', 'Laki-laki']:
        if klas == 'Dengan Risiko':
            saran = "Modifikasi gaya hidup. Pria cenderung menumpuk lemak di perut, jaga pola makan untuk menghindari risiko penyakit jantung."
        elif klas == 'Obesitas I':
            saran = "Risiko tinggi. Khusus pria, waspadai penumpukan lemak perut (visceral fat). Lakukan diet dan kardio teratur."
        elif klas == 'Obesitas II':
            saran = "Risiko sangat tinggi! Segera evaluasi klinis kardiovaskular karena pria lebih rentan komplikasi jantung akibat obesitas."
    else: # Perempuan
        if klas == 'Dengan Risiko':
            saran = "Modifikasi gaya hidup. Mulai rutinkan senam atau aerobik untuk menjaga keseimbangan dan kebugaran tubuh."
        elif klas == 'Obesitas I':
            saran = "Risiko tinggi. Perhatikan asupan gizi dan keseimbangan hormon. Lakukan aktivitas fisik secara teratur."
        elif klas == 'Obesitas II':
            saran = "Risiko sangat tinggi! Lakukan evaluasi klinis medis menyeluruh dan pertimbangkan terapi spesifik."
            
    imt_val = bb / ((tb / 100) ** 2) if tb > 0 else 0

    result = dict(
        imt_val=round(imt_val, 1), klas=klas,
        hasil=crisp, diag=diag, saran=saran, level=level,
        charts=charts, steps=steps,
        umur=umur, jk=jk, bb=bb, tb=tb,
    )
    return render_template('index.html', page='demo', result=result)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
