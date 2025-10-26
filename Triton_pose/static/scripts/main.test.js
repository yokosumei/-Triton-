(() => {
  const qs = (s) => document.querySelector(s);
  const btnStart = qs('#btn-start');
  const btnStop  = qs('#btn-stop');
  const statusEl = qs('#status');
  const labelEl  = qs('#label');
  const probaEl  = qs('#proba');
  const rawImg   = qs('#raw');
  const poseImg  = qs('#pose');

  let socket;
  let rafId = null;
  const throttleMs = 100; // evitam spam-ul de DOM updates

  function safeUpdate(cb) {
    if (rafId) return;
    rafId = requestAnimationFrame(() => {
      cb();
      rafId = null;
    });
  }

  function refreshFeeds() {
    const ts = Date.now();
    rawImg.src  = '/video_feed?ts='  + ts;  // anti-cache burst
    poseImg.src = '/pose_feed?ts=' + ts;
  }

  btnStart.onclick = async () => {
    await fetch('/start', { method: 'POST' });
    statusEl.textContent = 'running';
    refreshFeeds();

    if (!socket) {
      socket = io();
      socket.on('decision', (m) => {
        const { label = 'N/A', proba = 0 } = m || {};
        // throttle mic pentru DOM:
        setTimeout(() => safeUpdate(() => {
          labelEl.textContent = label;
          probaEl.textContent = Number(proba).toFixed(3);
        }), throttleMs);
      });
    }
  };

  btnStop.onclick = async () => {
    await fetch('/stop', { method: 'POST' });
    statusEl.textContent = 'stopped';
  };
})();
