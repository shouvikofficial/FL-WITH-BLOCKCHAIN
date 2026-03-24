/**
 * client_dashboard.js
 * ===================
 * Polls /api/client_log?client_id=<id> every 2s.
 * The client_id is read from the URL: ?client_id=Client_1
 * Each client only ever sees their own data.
 */

// ============================================================
// READ CLIENT ID FROM URL
// ============================================================
const urlParams   = new URLSearchParams(window.location.search);
const CLIENT_ID   = urlParams.get('client_id') || '';

// If no client_id in URL — show a picker instead of the dashboard
if (!CLIENT_ID) {
    showClientPicker();
}

async function showClientPicker() {
    document.querySelector('.main').innerHTML = `
        <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:100%; gap:24px; padding:40px;">
            <div style="font-size:2rem;">⚡</div>
            <h2 style="color:#fff; font-size:1.4rem;">Select a Client to Monitor</h2>
            <p style="color:#6b82a8; font-size:0.9rem;">Choose which federated node you want to observe.</p>
            <div id="client-picker-list" style="display:flex; flex-direction:column; gap:12px; min-width:280px;">
                <div style="color:#6b82a8; font-size:0.85rem;">Loading active sessions...</div>
            </div>
        </div>`;
    try {
        const res  = await fetch('/api/client_sessions');
        const data = await res.json();
        const listEl = document.getElementById('client-picker-list');
        if (data.clients.length === 0) {
            listEl.innerHTML = '<div style="color:#6b82a8;">No active client sessions yet. Run real_client.py first.</div>';
        } else {
            listEl.innerHTML = data.clients.map(cid => `
                <a href="/client_dashboard?client_id=${encodeURIComponent(cid)}"
                   style="display:block; padding:16px 24px; border-radius:14px;
                          border:1px solid rgba(245,158,11,0.3); background:rgba(245,158,11,0.07);
                          color:#f59e0b; font-weight:600; text-decoration:none; text-align:center;
                          font-size:1rem; transition:all 0.2s;"
                   onmouseover="this.style.background='rgba(245,158,11,0.15)'"
                   onmouseout="this.style.background='rgba(245,158,11,0.07)'">
                    ⚡ ${cid}
                </a>`).join('');
        }
    } catch(e) {
        document.getElementById('client-picker-list').innerHTML =
            '<div style="color:#f87171;">Could not reach server.</div>';
    }
}

// Only run dashboard logic if we have a client_id
if (CLIENT_ID) {

// ============================================================
// CHART SETUP
// ============================================================
const ctx = document.getElementById('clientChart').getContext('2d');
Chart.defaults.color = '#6b82a8';
Chart.defaults.font.family = "'Inter', sans-serif";

const gradAcc = ctx.createLinearGradient(0, 0, 0, 220);
gradAcc.addColorStop(0, 'rgba(245, 158, 11, 0.4)');
gradAcc.addColorStop(1, 'rgba(245, 158, 11, 0.0)');

const gradF1 = ctx.createLinearGradient(0, 0, 0, 220);
gradF1.addColorStop(0, 'rgba(167, 139, 250, 0.35)');
gradF1.addColorStop(1, 'rgba(167, 139, 250, 0.0)');

const chart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            {
                label: 'Val Accuracy',
                data: [],
                borderColor: '#f59e0b',
                backgroundColor: gradAcc,
                borderWidth: 2,
                pointBackgroundColor: '#fff',
                pointBorderColor: '#f59e0b',
                pointRadius: 4,
                fill: true,
                tension: 0.4,
                yAxisID: 'yAcc'
            },
            {
                label: 'F1 Score',
                data: [],
                borderColor: '#a78bfa',
                backgroundColor: gradF1,
                borderWidth: 2,
                pointBackgroundColor: '#fff',
                pointBorderColor: '#a78bfa',
                pointRadius: 4,
                fill: true,
                tension: 0.4,
                yAxisID: 'yAcc'
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
            legend: {
                position: 'top',
                labels: { boxWidth: 10, padding: 16, font: { size: 11 } }
            },
            tooltip: {
                callbacks: {
                    label: ctx => `${ctx.dataset.label}: ${(ctx.parsed.y * 100).toFixed(2)}%`
                }
            }
        },
        scales: {
            yAcc: {
                type: 'linear',
                position: 'left',
                min: 0, max: 1,
                title: { display: true, text: 'Score', color: '#f59e0b', font: { size: 10 } },
                grid: { color: 'rgba(255,255,255,0.04)' },
                ticks: { callback: v => (v * 100).toFixed(0) + '%', font: { size: 10 } }
            },
            x: {
                grid: { display: false },
                ticks: { font: { size: 10 } }
            }
        }
    }
});

// ============================================================
// STATE
// ============================================================
let lastEventCount = 0;
let derivedState = {
    clientId:     null,
    serverUrl:    null,
    totalRounds:  null,
    currentRound: 0,
    phase:        'idle',
    samples:      null,
    features:     null,
    scalerType:   null,
    smote:        null,
    poison:       null,
    classDist:    null,
    samplesSMOTE: null,
    // Latest training metrics
    localAcc:   null,
    trainAcc:   null,
    f1:         null,
    precision:  null,
    recall:     null,
    rocAuc:     null,
    nIter:      null,
    duration:   null,
    // Chart data
    accData: [],
    f1Data:  [],
};


const EVENT_SYMBOLS = {
    session_start:  ['🚀', 'session_start'],
    data_loaded:    ['📦', 'data_loaded'],
    preprocess:     ['🔬', 'preprocess'],
    round_start:    ['🔄', 'round_start'],
    global_model:   ['📡', 'global_model'],
    training_done:  ['🧠', 'training_done'],
    submit_done:    ['📤', 'submit_done'],
    waiting:        ['⏳', 'waiting'],
    round_complete: ['✅', 'round_complete'],
    session_end:    ['🏁', 'session_end'],
    error:          ['❌', 'error'],
};

// ============================================================
// POLLING
// ============================================================
async function poll() {
    try {
        const res  = await fetch(`/api/client_log?client_id=${encodeURIComponent(CLIENT_ID)}&t=` + Date.now());
        const data = await res.json();
        if (data.events && data.events.length !== lastEventCount) {
            lastEventCount = data.events.length;
            processEvents(data.events);
            renderUI();
        }
    } catch (e) {
        setConnStatus('error', 'Cannot reach server');
    }
}

setInterval(poll, 2000);
poll();   // immediate first load

// ============================================================
// EVENT PROCESSING
// ============================================================
function processEvents(events) {
    // Reset derived state per session, preserving chart history
    let hasSession = false;

    events.forEach(ev => {
        switch (ev.type) {
            case 'session_start':
                if (!hasSession) {
                    // New session detected → reset chart
                    derivedState.accData  = [];
                    derivedState.timeData = [];
                    derivedState.currentRound = 0;
                    hasSession = true;
                }
                derivedState.clientId    = ev.client_id;
                derivedState.serverUrl   = ev.server_url;
                derivedState.totalRounds = ev.total_rounds;
                derivedState.phase       = 'loading';
                break;

            case 'data_loaded':
                derivedState.samples  = ev.num_samples;
                derivedState.features = ev.num_features;
                derivedState.classDist = ev.class_distribution;
                derivedState.phase    = 'loading';
                break;

            case 'preprocess':
                derivedState.scalerType  = ev.scaler_type;
                derivedState.smote       = ev.smote_applied;
                derivedState.poison      = ev.poisoning_detected;
                derivedState.samplesSMOTE = ev.samples_after_smote;
                break;

            case 'round_start':
                derivedState.currentRound = ev.round;
                derivedState.phase = 'training';
                break;

            case 'training_done':
                derivedState.localAcc  = ev.local_accuracy;
                derivedState.trainAcc  = ev.train_accuracy;
                derivedState.f1        = ev.f1_score;
                derivedState.precision = ev.precision;
                derivedState.recall    = ev.recall;
                derivedState.rocAuc    = ev.roc_auc;
                derivedState.nIter     = ev.n_iter;
                derivedState.duration  = ev.duration_sec;
                const lbl = `Rnd ${ev.round}`;
                if (!chart.data.labels.includes(lbl)) {
                    chart.data.labels.push(lbl);
                    derivedState.accData.push(ev.local_accuracy);
                    derivedState.f1Data.push(ev.f1_score ?? ev.local_accuracy);
                }
                break;

            case 'submit_done':
                derivedState.phase = 'waiting';
                break;

            case 'waiting':
                derivedState.phase = 'waiting';
                break;

            case 'round_complete':
                if (ev.result === 'all_done') {
                    derivedState.phase = 'complete';
                } else {
                    derivedState.phase = 'training';
                }
                break;

            case 'session_end':
                derivedState.phase = 'complete';
                break;

            case 'error':
                derivedState.phase = 'error';
                break;
        }
    });

    // Rebuild feed
    buildFeed(events);
}

// ============================================================
// RENDER UI
// ============================================================
function renderUI() {
    const s = derivedState;

    // Sidebar
    if (s.clientId)    document.getElementById('sidebar-client-id').textContent = s.clientId;
    if (s.serverUrl)   document.getElementById('sidebar-server-url').textContent = s.serverUrl;
    if (s.totalRounds) document.getElementById('sidebar-total-rounds').textContent = s.totalRounds;
    document.getElementById('sidebar-round').textContent = s.currentRound || '0';

    // Progress bar
    const pct = s.totalRounds ? Math.min(100, (s.currentRound / s.totalRounds) * 100) : 0;
    document.getElementById('round-progress-fill').style.width = pct + '%';

    // Phase badge
    renderPhaseBadge(s.phase);

    // Stat cards
    if (s.samplesSMOTE !== null) {
        document.getElementById('sc-samples').textContent = s.samplesSMOTE.toLocaleString();
    } else if (s.samples !== null) {
        document.getElementById('sc-samples').textContent = s.samples.toLocaleString();
    }
    if (s.features !== null) document.getElementById('sc-features').textContent = s.features;
    if (s.localAcc !== null) {
        const pct = (s.localAcc * 100).toFixed(2) + '%';
        const f1Txt = s.f1 !== null ? ` · F1: ${(s.f1*100).toFixed(1)}%` : '';
        document.getElementById('sc-acc').textContent = pct;
        document.getElementById('sc-acc').title = `Val Acc: ${pct}${f1Txt}`;
    }
    if (s.duration !== null) document.getElementById('sc-time').textContent = s.duration + 's';

    // Preprocessing panel
    renderPreproc(s);

    // Stepper
    renderStepper(s);

    // Chart
    chart.data.datasets[0].data = derivedState.accData;
    chart.data.datasets[1].data = derivedState.f1Data;
    chart.update('none');

    // Connection status
    if (s.phase === 'idle') {
        setConnStatus('idle', 'No active session');
    } else if (s.phase === 'error') {
        setConnStatus('error', 'Connection error');
    } else if (s.phase === 'complete') {
        setConnStatus('idle', 'Session complete');
    } else {
        setConnStatus('connected', `${s.clientId || 'Client'} · Round ${s.currentRound}`);
    }

    // Round stepper label
    document.getElementById('round-stepper-label').textContent =
        `Round ${s.currentRound || '—'} / ${s.totalRounds || '—'}`;
}

// ============================================================
// PHASE BADGE
// ============================================================
const PHASE_LABELS = {
    idle:     '⬤ Idle',
    loading:  '⬤ Loading',
    training: '⬤ Training',
    waiting:  '⬤ Waiting',
    complete: '⬤ Complete',
    error:    '⬤ Error',
};

function renderPhaseBadge(phase) {
    const el = document.getElementById('phase-badge');
    el.className = `phase-badge ${phase}`;
    el.textContent = PHASE_LABELS[phase] || '⬤ Idle';
}

// ============================================================
// PREPROCESSING PANEL
// ============================================================
function renderPreproc(s) {
    if (s.currentRound > 0) {
        document.getElementById('preproc-round-badge').textContent = `Round ${s.currentRound}`;
    }

    if (s.scalerType !== null) {
        const scalerEl = document.getElementById('pp-scaler');
        scalerEl.innerHTML = s.scalerType === 'global'
            ? '<span class="pill teal">🌐 Global</span>'
            : '<span class="pill amber">🏠 Local</span>';
    }

    if (s.smote !== null) {
        document.getElementById('pp-smote').innerHTML = s.smote
            ? '<span class="pill on">✓ Applied</span>'
            : '<span class="pill off">— Not Needed</span>';
    }

    if (s.poison !== null) {
        document.getElementById('pp-poison').innerHTML = s.poison
            ? '<span class="pill warn">⚠ Detected</span>'
            : '<span class="pill on">✓ Clean</span>';
    }

    if (s.samplesSMOTE !== null) {
        document.getElementById('pp-samples').innerHTML =
            `<span style="color:#fff">${s.samplesSMOTE.toLocaleString()}</span>`;
    }

    // Class distribution bars
    if (s.classDist) {
        const total = Object.values(s.classDist).reduce((a, b) => a + b, 0);
        const colors = ['#f59e0b', '#14b8a6', '#a78bfa', '#60a5fa'];
        const labels = { '0': 'Class 0 (No Disease)', '1': 'Class 1 (Disease)' };
        const barsEl = document.getElementById('class-dist-bars');
        barsEl.innerHTML = '';
        Object.entries(s.classDist).forEach(([k, count], i) => {
            const pct = total > 0 ? (count / total * 100).toFixed(1) : '0.0';
            const color = colors[i % colors.length];
            barsEl.innerHTML += `
                <div class="class-bar-wrap">
                    <div class="class-bar-label">
                        <span>${labels[k] || `Class ${k}`}</span>
                        <span style="color:${color}">${count} (${pct}%)</span>
                    </div>
                    <div class="class-bar-bg">
                        <div class="class-bar-fill" style="width:${pct}%; background:${color}"></div>
                    </div>
                </div>`;
        });
    }
}

// ============================================================
// STEPPER
// ============================================================
function renderStepper(s) {
    const steps = ['load', 'preprocess', 'global', 'train', 'submit', 'wait'];
    const phaseMap = {
        idle:     0,
        loading:  1,
        training: 3,
        waiting:  5,
        complete: 6,
        error:    0,
    };

    const activeIdx = phaseMap[s.phase] ?? 0;

    // Step 1: data load — done if we have samples
    const step1done = s.samples !== null;

    steps.forEach((id, i) => {
        const el = document.getElementById(`step-${id}`);
        if (!el) return;
        el.className = 'step-item';
        if (i < activeIdx || (i === 0 && step1done)) {
            el.classList.add('done');
        } else if (i === activeIdx) {
            el.classList.add('active');
        }
    });

    // Step detail updates
    if (s.samples !== null) {
        document.getElementById('step-detail-load').textContent =
            `${s.samples.toLocaleString()} samples · ${s.features} features`;
    }
    if (s.scalerType !== null) {
        document.getElementById('step-detail-preprocess').textContent =
            `${s.scalerType} scaler · SMOTE: ${s.smote ? 'Yes' : 'No'} · Poison: ${s.poison ? 'Yes' : 'No'}`;
    }
    if (s.localAcc !== null) {
        const f1txt  = s.f1   !== null ? ` · F1: ${(s.f1*100).toFixed(2)}%`    : '';
        const rottxt = s.rocAuc !== null ? ` · AUC: ${(s.rocAuc*100).toFixed(2)}%` : '';
        const itertxt = s.nIter !== null ? ` · iters: ${s.nIter}` : '';
        document.getElementById('step-detail-train').textContent =
            `Val: ${(s.localAcc*100).toFixed(2)}%${f1txt}${rottxt} · ${s.duration}s${itertxt}`;
    }
    if (s.phase === 'complete') {
        document.getElementById('step-detail-wait').textContent = 'Aggregation complete ✓';
    }
}

// ============================================================
// EVENT FEED
// ============================================================
function buildFeed(events) {
    const feed = document.getElementById('event-feed');
    document.getElementById('event-count').textContent = `${events.length} events`;

    const lines = events.map(ev => {
        const ts = new Date(ev.timestamp * 1000).toLocaleTimeString('en-GB');
        const [sym, cls] = EVENT_SYMBOLS[ev.type] || ['•', 'idle'];
        const msg = formatEventMsg(ev);
        return `<div class="feed-line">
            <span class="feed-ts">${ts}</span>
            <span class="feed-sym">${sym}</span>
            <span class="feed-msg ${cls}">${msg}</span>
        </div>`;
    });

    feed.innerHTML = lines.join('') || '<div class="feed-line"><span class="feed-ts">--:--:--</span><span class="feed-sym">◌</span><span class="feed-msg idle">Waiting for real_client.py to start...</span></div>';
    feed.scrollTop = feed.scrollHeight;
}

function formatEventMsg(ev) {
    switch (ev.type) {
        case 'session_start':
            return `Session started · ${ev.client_id} → ${ev.server_url} · ${ev.total_rounds} rounds`;
        case 'data_loaded':
            return `Loaded ${ev.num_samples} samples · ${ev.num_features} features · label: ${ev.label}`;
        case 'preprocess':
            return `Round ${ev.round} · scaler: ${ev.scaler_type} · SMOTE: ${ev.smote_applied} · poison: ${ev.poisoning_detected} · ${ev.samples_after_smote} samples`;
        case 'round_start':
            return `Round ${ev.round} / ${ev.total_rounds} started`;
        case 'global_model':
            return `Global model ${ev.received ? 'received ✓' : 'not yet available (round 1)'}`;
        case 'training_done': {
            const acc  = (ev.local_accuracy * 100).toFixed(2);
            const tacc = ev.train_accuracy ? ` · Train: ${(ev.train_accuracy*100).toFixed(2)}%` : '';
            const f1   = ev.f1_score   !== undefined && ev.f1_score !== null   ? ` · F1: ${(ev.f1_score*100).toFixed(2)}%`   : '';
            const auc  = ev.roc_auc    !== undefined && ev.roc_auc  !== null   ? ` · AUC: ${(ev.roc_auc*100).toFixed(2)}%`   : '';
            const itr  = ev.n_iter     !== undefined ? ` · ${ev.n_iter} iters` : '';
            return `Val Acc: ${acc}%${tacc}${f1}${auc} · ${ev.duration_sec}s${itr}`;
        }
        case 'submit_done':
            return `Submitted · status: ${ev.status} · ${ev.submitted}/${ev.expected} clients ready`;
        case 'waiting':
            return `Waiting for other clients to finish round ${ev.round}...`;
        case 'round_complete':
            return `Round ${ev.round} complete · result: ${ev.result}`;
        case 'session_end':
            return `Session ended · ${ev.rounds_done} rounds done · status: ${ev.status}`;
        case 'error':
            return `Error: ${ev.message}`;
        default:
            return JSON.stringify(ev);
    }
}

// ============================================================
// CONNECTION STATUS
// ============================================================
function setConnStatus(type, text) {
    const el   = document.getElementById('conn-status');
    const txtEl= document.getElementById('conn-text');
    el.className = `conn-status ${type}`;
    txtEl.textContent = text;
}

} // end if (CLIENT_ID)
