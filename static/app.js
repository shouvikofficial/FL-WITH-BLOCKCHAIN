document.addEventListener('DOMContentLoaded', () => {
    
    const startBtn = document.getElementById('start-btn');
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
    const terminalPane = document.getElementById('terminal-pane');
    const toggleTerminalBtn = document.getElementById('toggle-terminal-btn');

    toggleTerminalBtn.addEventListener('click', () => {
        terminalPane.classList.toggle('hidden');
        if (terminalPane.classList.contains('hidden')) {
            toggleTerminalBtn.innerText = 'Show Terminal Logs';
        } else {
            toggleTerminalBtn.innerText = 'Hide Terminal Logs';
        }
    });

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
        startBtn.innerText = 'Federation Active...';
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
        }
    });

    // Fetch and Update Logic
    async function fetchMetrics() {
        try {
            const res = await fetch('/api/metrics');
            const data = await res.json();

            // Status updates
            statusText.innerText = data.status || "Syncing...";
            if (data.round && data.round > 0) {
                currentRoundEl.innerText = data.round;
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

            // Attack Display
            if (data.malicious_attackers && data.malicious_attackers.length > 0) {
                attackBox.classList.remove('hidden');
                attackText.innerHTML = `Identified: <b>${data.malicious_attackers.join(', ')}</b><br>Defensive Median Aggregation Active.`;
            } else {
                attackBox.classList.add('hidden');
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
                clearInterval(pollingInterval);
                pollingInterval = null;
                startBtn.disabled = false;
                startBtn.innerText = 'Restart Federation';
                pulseDot.classList.remove('active');
                pulseDot.style.backgroundColor = '#6366f1';
                statusText.innerText = 'Training Complete';
            }

        } catch (e) {
            console.warn("Polling error (might be initial write): ", e);
        }
    }

    // Ping once on load to see if a session is already there
    fetchMetrics();
});
