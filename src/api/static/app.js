/**
 * QuantumSentiment LAN Dashboard — client logic
 */

const STORAGE_KEY = 'qs_dashboard_api_key';
const OLLAMA_MODEL_KEY = 'qs_dashboard_ollama_model';

function getApiKey() {
  return localStorage.getItem(STORAGE_KEY) || '';
}

function setApiKey(key) {
  localStorage.setItem(STORAGE_KEY, key);
}

function getOllamaModel() {
  return localStorage.getItem(OLLAMA_MODEL_KEY) || '';
}

function setOllamaModel(model) {
  if (model) localStorage.setItem(OLLAMA_MODEL_KEY, model);
  else localStorage.removeItem(OLLAMA_MODEL_KEY);
}

async function apiGet(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function apiPost(path, body) {
  const res = await fetch(path, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': getApiKey(),
    },
    body: JSON.stringify(body),
  });
  const text = await res.text();
  let data = {};
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    data = { detail: text };
  }
  if (!res.ok) {
    const detail = data.detail;
    const msg = typeof detail === 'string' && detail
      ? detail
      : Array.isArray(detail)
        ? detail.map((d) => d.msg || d).join('; ')
        : res.statusText;
    throw new Error(msg);
  }
  return data;
}

function fmtMoney(v) {
  if (v == null || Number.isNaN(v)) return '—';
  const sign = v >= 0 ? '' : '-';
  return sign + '$' + Math.abs(v).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

/** Format a decimal ratio as a percentage (0.005 → +0.50%, 0.15 → +15.00%). */
function fmtPct(v) {
  if (v == null || Number.isNaN(v)) return '—';
  const pct = v * 100;
  return (pct >= 0 ? '+' : '') + pct.toFixed(2) + '%';
}

function fmtTime(iso) {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

window.fmtMoney = fmtMoney;
window.fmtPct = fmtPct;
window.fmtTime = fmtTime;

let equityChart = null;
let pnlChart = null;

function renderPnlChart(canvasId, days) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const withPnl = (days || []).filter((d) => d.pnl != null);
  if (!withPnl.length) {
    if (pnlChart) {
      pnlChart.destroy();
      pnlChart = null;
    }
    return;
  }
  const labels = withPnl.map((d) => d.date);
  const data = withPnl.map((d) => d.pnl);
  const colors = data.map((v) =>
    v >= 0 ? 'rgba(52, 211, 153, 0.75)' : 'rgba(248, 113, 113, 0.75)'
  );
  if (pnlChart) pnlChart.destroy();
  pnlChart = new Chart(canvas, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Daily P&L',
        data,
        backgroundColor: colors,
        borderWidth: 0,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: '#94a3b8', maxTicksLimit: 8 }, grid: { display: false } },
        y: {
          ticks: { color: '#94a3b8', callback: (v) => '$' + v.toLocaleString() },
          grid: { color: 'rgba(148,163,184,0.1)' },
        },
      },
    },
  });
}

function renderEquityChart(canvasId, series) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !series || !series.length) return;
  const labels = series.map((p) => p.timestamp);
  const data = series.map((p) => p.equity);
  if (equityChart) equityChart.destroy();
  equityChart = new Chart(canvas, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Equity',
        data,
        borderColor: '#34d399',
        backgroundColor: 'rgba(52, 211, 153, 0.1)',
        fill: true,
        tension: 0.2,
        pointRadius: 0,
        borderWidth: 2,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: '#94a3b8', maxTicksLimit: 6 }, grid: { color: 'rgba(148,163,184,0.1)' } },
        y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(148,163,184,0.1)' } },
      },
    },
  });
}

function dashboardApp() {
  return {
    pnlPeriodOptions: [
      { k: '1M', l: '1M' },
      { k: '3M', l: '3M' },
      { k: '6M', l: '6M' },
      { k: '1A', l: '1Y' },
      { k: 'all', l: 'All' },
    ],
    tab: 'overview',
    loading: false,
    toast: null,
    apiKey: getApiKey(),
    refreshInterval: 5,
    overview: null,
    positions: [],
    orders: [],
    models: [],
    botStatus: null,
    backtestRuns: [],
    backtestForm: {
      symbols: 'AAPL',
      start_date: '2024-06-05',
      end_date: '2024-06-30',
      capital: 10000,
    },
    activeBacktestId: null,
    activeBacktest: null,
    pollTimer: null,
    backtestPollTimer: null,
    assistantStatus: null,
    ollamaModels: [],
    selectedOllamaModel: getOllamaModel(),
    chatMessages: [],
    chatInput: '',
    chatLoading: false,
    pnlHistory: null,
    pnlPeriod: '3M',
    pnlLoading: false,
    features: [],
    featuresSaving: false,
    riskSettings: null,
    riskSaving: false,

    tabLabel(t) {
      if (t === 'guide') return 'Help';
      if (t === 'assistant') return 'Assistant';
      if (t === 'pnl') return 'P&L';
      if (t === 'risk') return 'Risk';
      return t;
    },

    selectTab(t) {
      this.tab = t;
      if (t === 'assistant') this.loadAssistantStatus();
      if (t === 'pnl') this.loadPnlHistory();
      if (t === 'risk') this.loadRiskSettings();
    },

    init() {
      this.refreshInterval = 5;
      this.refreshAll();
      this.pollTimer = setInterval(() => {
        if (this.tab === 'overview') this.loadOverview();
        if (this.tab === 'positions') this.loadPositions();
        if (this.tab === 'activity') this.loadActivity();
      }, this.refreshInterval * 1000);
    },

    showToast(msg, type = 'info') {
      this.toast = { msg, type };
      setTimeout(() => { this.toast = null; }, 4000);
    },

    saveApiKey() {
      setApiKey(this.apiKey);
      this.showToast('API key saved locally', 'success');
    },

    async refreshAll() {
      this.loading = true;
      try {
        const health = await apiGet('/api/health');
        this.refreshInterval = health.refresh_interval_seconds || 5;
        await Promise.all([
          this.loadOverview(),
          this.loadPositions(),
          this.loadActivity(),
          this.loadModels(),
          this.loadBacktests(),
          this.loadFeatures(),
          this.loadRiskSettings(),
        ]);
      } catch (e) {
        this.showToast('Failed to load dashboard: ' + e.message, 'error');
      } finally {
        this.loading = false;
      }
    },

    async loadOverview() {
      try {
        this.overview = await apiGet('/api/overview');
        this.botStatus = this.overview?.bot;
      } catch (e) {
        console.error(e);
      }
    },

    async loadFeatures() {
      try {
        const data = await apiGet('/api/features');
        this.features = data.features || [];
      } catch (e) {
        console.error(e);
      }
    },

    async toggleFeature(feature, checked) {
      if (!feature.available) return;
      this.featuresSaving = true;
      try {
        const data = await apiPost('/api/features', { enabled: { [feature.id]: checked } });
        this.features = data.features || [];
        this.showToast(
          checked ? `${feature.label} enabled` : `${feature.label} disabled`,
          'success'
        );
      } catch (e) {
        this.showToast('Failed to update feature: ' + e.message, 'error');
        await this.loadFeatures();
      } finally {
        this.featuresSaving = false;
      }
    },

    async loadRiskSettings() {
      try {
        this.riskSettings = await apiGet('/api/risk');
      } catch (e) {
        console.error(e);
      }
    },

    riskParamLabel(key) {
      const val = this.riskSettings?.params?.[key];
      if (val == null || Number.isNaN(val)) return '—';
      if (key === 'max_leverage') return val.toFixed(1) + '×';
      return (val * 100).toFixed(1) + '%';
    },

    async applyRiskPreset(presetId) {
      this.riskSaving = true;
      try {
        this.riskSettings = await apiPost('/api/risk', { preset: presetId });
        const label = (this.riskSettings.presets || []).find((p) => p.id === presetId)?.label || presetId;
        this.showToast(`${label} preset applied`, 'success');
        if (this.tab === 'positions') await this.loadPositions();
      } catch (e) {
        this.showToast('Failed to apply preset: ' + e.message, 'error');
        await this.loadRiskSettings();
      } finally {
        this.riskSaving = false;
      }
    },

    async updateRiskParam(key, value) {
      if (value == null || Number.isNaN(value)) return;
      this.riskSaving = true;
      try {
        this.riskSettings = await apiPost('/api/risk', { params: { [key]: value } });
        if (this.tab === 'positions') await this.loadPositions();
      } catch (e) {
        this.showToast('Failed to update risk setting: ' + e.message, 'error');
        await this.loadRiskSettings();
      } finally {
        this.riskSaving = false;
      }
    },

    botHealthy() {
      return this.botStatus?.healthy === true;
    },

    botRunning() {
      return this.botStatus?.running === true;
    },

    async loadPositions() {
      try {
        const data = await apiGet('/api/positions');
        this.positions = data.positions || [];
      } catch (e) {
        console.error(e);
      }
    },

    async loadActivity() {
      try {
        const data = await apiGet('/api/orders');
        this.orders = data.orders || [];
      } catch (e) {
        console.error(e);
      }
    },

    async loadPnlHistory() {
      this.pnlLoading = true;
      try {
        this.pnlHistory = await apiGet('/api/pnl/history?period=' + encodeURIComponent(this.pnlPeriod));
        this.$nextTick(() => {
          renderPnlChart('pnlChart', this.pnlHistory?.days || []);
        });
      } catch (e) {
        this.pnlHistory = { error: e.message, days: [], summary: {} };
        this.showToast('Failed to load P&L history: ' + e.message, 'error');
      } finally {
        this.pnlLoading = false;
      }
    },

    setPnlPeriod(period) {
      this.pnlPeriod = period;
      this.loadPnlHistory();
    },

    pnlDaysNewestFirst() {
      return [...(this.pnlHistory?.days || [])].reverse();
    },

    async loadModels() {
      try {
        const data = await apiGet('/api/models');
        this.models = data.models || [];
      } catch (e) {
        console.error(e);
      }
    },

    async loadBacktests() {
      try {
        const data = await apiGet('/api/backtests');
        this.backtestRuns = data.runs || [];
      } catch (e) {
        console.error(e);
      }
    },

    async startBot() {
      try {
        const res = await apiPost('/api/bot/start', { mode: 'paper' });
        if (res.ok === false) throw new Error(res.error || 'Start failed');
        this.showToast('Paper trading bot started', 'success');
        await this.loadOverview();
      } catch (e) {
        this.showToast(e.message, 'error');
      }
    },

    async stopBot() {
      try {
        await apiPost('/api/bot/stop', {});
        this.showToast('Bot stop requested', 'success');
        await this.loadOverview();
      } catch (e) {
        this.showToast(e.message, 'error');
      }
    },

    async runBacktest() {
      try {
        const symbols = this.backtestForm.symbols.split(/[\s,]+/).filter(Boolean);
        const res = await apiPost('/api/backtests', {
          symbols,
          start_date: this.backtestForm.start_date,
          end_date: this.backtestForm.end_date,
          capital: Number(this.backtestForm.capital),
        });
        this.activeBacktestId = res.run_id;
        this.showToast('Backtest started: ' + res.run_id, 'success');
        this.pollBacktest(res.run_id);
        await this.loadBacktests();
      } catch (e) {
        this.showToast(e.message, 'error');
      }
    },

    pollBacktest(runId) {
      if (this.backtestPollTimer) clearInterval(this.backtestPollTimer);
      this.backtestPollTimer = setInterval(async () => {
        try {
          const status = await apiGet('/api/backtests/' + runId + '/status');
          if (status.state === 'done' || status.state === 'error') {
            clearInterval(this.backtestPollTimer);
            this.backtestPollTimer = null;
            await this.viewBacktest(runId);
            await this.loadBacktests();
            this.showToast(
              status.state === 'done' ? 'Backtest complete' : 'Backtest failed',
              status.state === 'done' ? 'success' : 'error'
            );
          }
        } catch (e) {
          console.error(e);
        }
      }, 2000);
    },

    async viewBacktest(runId) {
      try {
        const run = await apiGet('/api/backtests/' + runId);
        this.activeBacktestId = runId;
        this.activeBacktest = run;
        this.tab = 'backtest';
        this.$nextTick(() => {
          const series = run.results?.equity_series || [];
          renderEquityChart('equityChart', series);
        });
      } catch (e) {
        this.showToast(e.message, 'error');
      }
    },

    accountEquity() {
      return this.overview?.account?.equity;
    },

    dailyPnl() {
      return this.overview?.account?.daily_pnl;
    },

    dailyPnlPct() {
      return this.overview?.account?.daily_pnl_percent;
    },

    async loadAssistantStatus() {
      try {
        this.assistantStatus = await apiGet('/api/assistant/status');
        this.ollamaModels = this.assistantStatus?.models || [];
        const saved = getOllamaModel();
        const serverDefault =
          this.assistantStatus?.default_model ||
          this.assistantStatus?.ollama_model ||
          'llama3.2:1b';
        if (saved && this.ollamaModels.includes(saved)) {
          this.selectedOllamaModel = saved;
        } else if (this.ollamaModels.includes(serverDefault)) {
          this.selectedOllamaModel = serverDefault;
        } else if (this.ollamaModels.length) {
          this.selectedOllamaModel = this.ollamaModels[0];
        } else {
          this.selectedOllamaModel = serverDefault;
        }
        setOllamaModel(this.selectedOllamaModel);
      } catch (e) {
        this.assistantStatus = { ollama_available: false };
        console.error(e);
      }
    },

    onOllamaModelChange() {
      setOllamaModel(this.selectedOllamaModel);
    },

    scrollChatToBottom() {
      this.$nextTick(() => {
        const el = document.getElementById('assistantChatScroll');
        if (el) el.scrollTop = el.scrollHeight;
      });
    },

    async sendChat() {
      const text = (this.chatInput || '').trim();
      if (!text || this.chatLoading) return;
      this.chatMessages.push({ role: 'user', content: text });
      this.chatInput = '';
      this.chatLoading = true;
      this.scrollChatToBottom();
      try {
        const history = this.chatMessages
          .slice(0, -1)
          .filter((m) => m.role === 'user' || m.role === 'assistant')
          .map((m) => ({ role: m.role, content: m.content }));
        const res = await apiPost('/api/assistant/chat', {
          message: text,
          history,
          refresh_context: true,
          model: this.selectedOllamaModel || undefined,
        });
        this.chatMessages.push({
          role: 'assistant',
          content: res.reply || '(empty response)',
        });
      } catch (e) {
        this.chatMessages.push({
          role: 'assistant',
          content: 'Error: ' + (e.message || 'request failed'),
        });
      } finally {
        this.chatLoading = false;
        this.scrollChatToBottom();
      }
    },
  };
}

document.addEventListener('alpine:init', () => {
  Alpine.data('dashboardApp', dashboardApp);
});
