(function () {
  const btnStart = document.getElementById('btn-start');
  const btnStop  = document.getElementById('btn-stop');
  const statusEl = document.getElementById('status');
  const labelEl  = document.getElementById('label');
  const probaEl  = document.getElementById('proba');

  const socket = io();

  btnStart.onclick = async () => {
    await fetch('/start', { method: 'POST' });
    statusEl.textContent = 'running';
    // forțează refresh-ul stream-ului (unele browsere cache-uiesc)
    document.getElementById('stream').src = '/video_feed?ts=' + Date.now();
  };

  btnStop.onclick = async () => {
    await fetch('/stop', { method: 'POST' });
    statusEl.textContent = 'stopped';
  };

  socket.on('decision', (msg) => {
    labelEl.textContent = msg.label || 'N/A';
    probaEl.textContent = (msg.proba ?? 0).toFixed(2);
  });
})();
