document.addEventListener('DOMContentLoaded', () => {
    
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const statusText = document.getElementById('status-text');
    const pulseDot = document.getElementById('pulse');
    const currentRoundEl = document.getElementById('current-round');
    
    // Stats
    const mAcc = document.getElementById('metric-acc');
    const mF1 = document.getElementById('metric-f1');
    const mRoc = document.getElementById('metric-roc');
    const mPrec = document.getElementById('metric-prec');

    // Attack
    const attackBox = document.getElementById('attack-log');
    // Terminal
    const terminalOut = document.getElementById('terminal-output');

    // SPA Navigation Logic
    const navBtns = document.querySelectorAll('.nav-btn');
    const views = document.querySelectorAll('.view-section');

    navBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Reset active states
            navBtns.forEach(b => {
                b.classList.remove('active');
                if (b.id === 'nav-dashboard') {
                    b.style.background = 'var(--glass-bg)'; 
                } else {
                    b.style.background = 'transparent';
                }
            });
            views.forEach(v => v.classList.add('hidden'));

            // Set clicked as active
            btn.classList.add('active');
            if (btn.id === 'nav-dashboard') {
                btn.style.background = 'linear-gradient(135deg, var(--primary), var(--secondary))';
            } else {
                btn.style.background = 'rgba(99, 102, 241, 0.1)';
            }

            // Show proper view
            const targetId = btn.getAttribute('data-target') || 'view-dashboard';
            document.getElementById(targetId).classList.remove('hidden');

            // Fetch specific data
            if (targetId === 'view-blockchain') {
                fetchBlockchainLogs();
            } else if (targetId === 'view-attack') {
                updateAttackAuditView(); // Update from already fetched info if possible, or fetch
            }
        });
    });

    // Default dashboard btn state
    const dashboardBtn = document.getElementById('nav-dashboard');
    if (dashboardBtn) {
        dashboardBtn.style.background = 'linear-gradient(135deg, var(--primary), var(--secondary))';
        dashboardBtn.style.border = 'none';
        dashboardBtn.style.color = 'white';
    }

    // Chart.js Setup
    const ctx = document.getElementById('metricsChart').getContext('2d');
    Chart.defaults.color = '#a1a1aa';
    Chart.defaults.font.family = "'Outfit', sans-serif";

    // Gradient for line
    let gradientAcc = ctx.createLinearGradient(0, 0, 0, 400);
    gradientAcc.addColorStop(0, 'rgba(99, 102, 241, 0.5)');   
    gradientAcc.addColorStop(1, 'rgba(99, 102, 241, 0.0)');

    let gradientF1 = ctx.createLinearGradient(0, 0, 0, 400);
    gradientF1.addColorStop(0, 'rgba(236, 72, 153, 0.5)');   
    gradientF1.addColorStop(1, 'rgba(236, 72, 153, 0.0)');

    const chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Accuracy',
                    data: [],
                    borderColor: '#6366f1',
                    backgroundColor: gradientAcc,
                    borderWidth: 3,
                    pointBackgroundColor: '#fff',
                    pointBorderColor: '#6366f1',
                    pointRadius: 4,
                    fill: true,
                    tension: 0.4
                },
                {
                    label: 'F1 Score',
                    data: [],
                    borderColor: '#ec4899',
                    backgroundColor: gradientF1,
                    borderWidth: 3,
                    pointBackgroundColor: '#fff',
                    pointBorderColor: '#ec4899',
                    pointRadius: 4,
                    fill: true,
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: { boxWidth: 12, padding: 20 }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1.0,
                    grid: { color: 'rgba(255, 255, 255, 0.05)' }
                },
                x: {
                    grid: { display: false }
                }
            }
        }
    });

    let pollingInterval = null;

    // Start Training API Call
    startBtn.addEventListener('click', async () => {
        startBtn.disabled = true;
        startBtn.classList.add('hidden');
        stopBtn.classList.remove('hidden');
        
        pulseDot.classList.add('active');
        statusText.innerText = 'Initializing Nodes...';

        try {
            await fetch('/api/start', { method: 'POST' });
            // Start polling if not already
            if (!pollingInterval) {
                pollingInterval = setInterval(fetchMetrics, 2000);
            }
        } catch (e) {
            console.error(e);
            statusText.innerText = 'Failed to Start';
            pulseDot.classList.remove('active');
            startBtn.disabled = false;
            startBtn.classList.remove('hidden');
            stopBtn.classList.add('hidden');
        }
    });

    // Stop Training API Call
    stopBtn.addEventListener('click', async () => {
        stopBtn.disabled = true;
        stopBtn.innerText = 'Stopping...';
        try {
            await fetch('/api/stop', { method: 'POST' });
            // The polling loop will naturally catch the "completed=true" from the backend, 
            // but we can fast-track UI updating here:
            if (pollingInterval) {
                clearInterval(pollingInterval);
                pollingInterval = null;
            }
            startBtn.disabled = false;
            startBtn.classList.remove('hidden');
            startBtn.innerText = 'Restart Federation';
            
            stopBtn.classList.add('hidden');
            stopBtn.disabled = false;
            stopBtn.innerText = 'Stop Training';
            
            pulseDot.classList.remove('active');
            pulseDot.style.backgroundColor = '#ef4444';
            statusText.innerText = 'Training Stopped';
            
            // fetch one last time to sync UI to stopped state
            fetchMetrics();
        } catch (e) {
            console.error(e);
            stopBtn.disabled = false;
            stopBtn.innerText = 'Stop Training';
        }
    });

    // Fetch and Update Logic
    async function fetchMetrics() {
        try {
            const res = await fetch('/api/metrics?t=' + Date.now());
            const data = await res.json();

            // Auto-start polling if an active session is detected and not already polling
            if (data.status && !data.status.includes('Waiting') && !data.status.includes('Stopped')) {
                if (!pollingInterval && !data.completed) {
                    pollingInterval = setInterval(fetchMetrics, 2000);
                    startBtn.classList.add('hidden');
                    stopBtn.classList.remove('hidden');
                    pulseDot.classList.add('active');
                }
            }

            // Status updates
            statusText.innerText = data.status || "Syncing...";
            if (data.round && data.round > 0) {
                currentRoundEl.innerText = data.round;
            }
            if (data.total_rounds) {
                const totalRoundsEl = document.getElementById('total-rounds');
                if (totalRoundsEl) totalRoundsEl.innerText = data.total_rounds;
            }

            // Update Metrics Cards if we have data
            if (data.metrics && data.metrics.accuracy && data.metrics.accuracy.length > 0) {
                const arrA = data.metrics.accuracy;
                const arrF = data.metrics.f1;
                const arrR = data.metrics.roc;
                const arrP = data.metrics.precision;

                mAcc.innerText = (arrA[arrA.length - 1] * 100).toFixed(2) + '%';
                mF1.innerText = arrF[arrF.length - 1].toFixed(4);
                mRoc.innerText = arrR[arrR.length - 1].toFixed(4);
                mPrec.innerText = arrP[arrP.length - 1].toFixed(4);

                // Update Chart
                chartInstance.data.labels = Array.from({length: arrA.length}, (_, i) => `Rnd ${i+1}`);
                chartInstance.data.datasets[0].data = arrA;
                chartInstance.data.datasets[1].data = arrF;
                chartInstance.update();
            }

            // Attack Display (Left Sidebar)
            if (data.malicious_attackers && data.malicious_attackers.length > 0) {
                attackBox.classList.remove('hidden');
                document.getElementById('attack-text').innerHTML = `Identified: <b>${data.malicious_attackers.join(', ')}</b><br>Defensive Median Aggregation Active.`;
            } else {
                attackBox.classList.add('hidden');
            }

            // Also update the full Attack Audit View automatically
            window.lastMetricsData = data;
            if (document.getElementById('view-attack') && !document.getElementById('view-attack').classList.contains('hidden')) {
                updateAttackAuditView(data);
            }

            // Also update the Blockchain logs automatically if viewing that tab
            if (document.getElementById('view-blockchain') && !document.getElementById('view-blockchain').classList.contains('hidden')) {
                fetchBlockchainLogs();
            }

            // Sync Terminal Stream
            try {
                const logRes = await fetch('/static/training_log.txt?t=' + Date.now());
                if (logRes.ok) {
                    const logText = await logRes.text();
                    terminalOut.innerText = logText;
                    terminalOut.scrollTop = terminalOut.scrollHeight; // Auto-scroll to bottom
                }
            } catch (err) {
                console.warn("Terminal fetch error:", err);
            }

            // Check if finished
            if (data.completed) {
                // Update Analytics View
                const analyticsPlaceholder = document.getElementById('analytics-placeholder');
                const analyticsGrid = document.getElementById('analytics-grid');
                if (analyticsPlaceholder && analyticsGrid) {
                    analyticsPlaceholder.classList.add('hidden');
                    analyticsGrid.classList.remove('hidden');
                    
                    const t = Date.now();
                    document.getElementById('img-shap').src = '/static/shap_summary.png?t=' + t;
                    document.getElementById('img-cm').src = '/static/confusion_matrix.png?t=' + t;
                    document.getElementById('img-roc').src = '/static/roc_curve.png?t=' + t;
                    document.getElementById('img-pr').src = '/static/pr_curve.png?t=' + t;
                }

                clearInterval(pollingInterval);
                pollingInterval = null;
                startBtn.disabled = false;
                startBtn.classList.remove('hidden');
                startBtn.innerText = 'Restart Federation';
                stopBtn.classList.add('hidden');
                
                pulseDot.classList.remove('active');
                if (data.status && data.status.includes('Stopped')) {
                    pulseDot.style.backgroundColor = '#ef4444';
                } else {
                    pulseDot.style.backgroundColor = '#6366f1';
                }
                statusText.innerText = data.status || 'Training Complete';
            }

        } catch (e) {
            console.warn("Polling error (might be initial write): ", e);
        }
    }

    // Backend fetches
    async function fetchBlockchainLogs() {
        const tbody = document.getElementById('logs-body');
        // Only show "Fetching" if the table is currently empty or showing the default placeholder
        if (tbody.children.length === 0 || tbody.innerHTML.includes("No updates") || tbody.innerHTML.includes("Failed")) {
            tbody.innerHTML = '<tr><td colspan="4" style="text-align:center; padding:2rem; color:#9ca3af;">Fetching securely from blockchain...</td></tr>';
        }
        try {
            const res = await fetch('/api/blockchain_logs?t=' + Date.now());
            const data = await res.json();
            
            if (data.success) {
                if (data.logs.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="4" style="text-align:center; padding:2rem; color:#9ca3af;">No updates logged on the blockchain yet.</td></tr>';
                } else {
                    tbody.innerHTML = '';
                    [...data.logs].reverse().forEach(log => {
                        const tr = document.createElement('tr');
                        const dateStr = new Date(log.timestamp * 1000).toLocaleString();
                        
                        tr.innerHTML = `
                            <td>${log.round}</td>
                            <td style="font-weight:600; color:#fff;">${log.client}</td>
                            <td style="font-size: 0.8rem; color:#9ca3af;">${dateStr}</td>
                            <td style="font-size: 0.8rem;">
                                <div style="margin-bottom:4px;" class="hash-cell" title="Model Hash: ${log.hash}">
                                    <span style="color:#6366f1;">MH:</span> ${log.hash.substring(0, 10)}...${log.hash.substring(log.hash.length - 6)}
                                </div>
                                <div class="hash-cell" title="Block Hash: ${log.block_hash}\nTx Hash: ${log.tx_hash}">
                                    <span style="color:#eab308;">BH:</span> ${log.block_hash ? log.block_hash.substring(0, 10) + '...' + log.block_hash.substring(log.block_hash.length - 6) : 'Unknown'}
                                </div>
                            </td>
                        `;
                        tbody.appendChild(tr);
                    });
                }
            }
        } catch (err) {
            console.error(err);
            tbody.innerHTML = '<tr><td colspan="4" style="text-align:center; padding:2rem; color:#ef4444;">Failed to fetch logs from server.</td></tr>';
        }
    }

    async function updateAttackAuditView(overrideData) {
        const contentDiv = document.getElementById('audit-content');
        let data = overrideData || window.lastMetricsData;
        
        if (!data) {
            try {
                const res = await fetch('/api/metrics');
                data = await res.json();
            } catch (err) {
                contentDiv.innerHTML = '<div style="text-align:center; padding:2rem; color:#ef4444;">Failed to fetch security audit.</div>';
                return;
            }
        }

        if (data.malicious_attackers && data.malicious_attackers.length > 0) {
            const attackersHtml = data.malicious_attackers.map(a => `<span class="attacker-badge">⚠️ ${a}</span>`).join('');
            contentDiv.innerHTML = `
                <div class="attack-card">
                    <h4 style="color: #ef4444; margin-top: 0; display: flex; align-items: center; gap: 0.5rem;">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
                        Protocol Violation Detected
                    </h4>
                    <p style="color: #d1d5db; margin-bottom: 1.5rem;">Anomalous weight updates deviating drastically from the global distribution were identified.</p>
                    <div style="margin-bottom: 1rem;">
                        <div style="font-size: 0.85rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem;">Identified Hostile Nodes</div>
                        ${attackersHtml}
                    </div>
                    <div class="defense-info">
                        <h5 style="color: #10b981; margin-top: 0;">🛡️ Active Countermeasures</h5>
                        <p style="color: #a7f3d0; margin-bottom: 0; font-size: 0.95rem;">
                            Malicious weight vectors have been isolated and dropped. 
                            The global model was securely aggregated using Trimmed Mean / Median Defense algorithms to preserve integrity.
                        </p>
                    </div>
                </div>
            `;
        } else {
            if (data.status && data.status.includes('Waiting')) {
                contentDiv.innerHTML = `
                    <div class="safe-card">
                        <h4 style="color: #9ca3af; margin-top:0;">System Idle</h4>
                        <p style="color: #6b7280; margin-bottom:0;">Awaiting federation initialization to begin threat monitoring.</p>
                    </div>
                `;
            } else {
                contentDiv.innerHTML = `
                    <div class="safe-card">
                        <svg style="color: #10b981; margin-bottom: 1rem;" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
                        <h4 style="color: #10b981; margin-top:0;">Zero Threats Detected</h4>
                        <p style="color: #34d399; margin-bottom:0; font-size: 0.95rem;">All nodes are operating within expected behavioral distribution patterns. Continual monitoring is active.</p>
                    </div>
                `;
            }
        }
    }

    // Ping once on load to see if a session is already there
    fetchMetrics();
});
