require('dotenv').config();
const http = require('http');
const fs = require('fs');
const path = require('path');
const https = require('https');
const os = require('os');
const { execFile } = require('child_process');

const PORT = process.env.PORT || 3000;
const API_KEY = process.env.ANTHROPIC_API_KEY;

if (!API_KEY) {
  console.error('ERROR: ANTHROPIC_API_KEY is not set in .env');
  process.exit(1);
}

// Try to find a working Python executable
const PYTHON = process.env.PYTHON || 'python';

function readBody(req) {
  return new Promise((resolve, reject) => {
    let body = '';
    req.on('data', chunk => { body += chunk; });
    req.on('end', () => resolve(body));
    req.on('error', reject);
  });
}

const server = http.createServer(async (req, res) => {

  // Serve index.html
  if (req.method === 'GET' && (req.url === '/' || req.url === '/index.html')) {
    const filePath = path.join(__dirname, 'index.html');
    fs.readFile(filePath, (err, data) => {
      if (err) { res.writeHead(500); res.end('Error loading page'); return; }
      res.writeHead(200, { 'Content-Type': 'text/html' });
      res.end(data);
    });
    return;
  }

  // Claude proxy
  if (req.method === 'POST' && req.url === '/api/messages') {
    const body = await readBody(req);
    const options = {
      hostname: 'api.anthropic.com',
      path: '/v1/messages',
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': API_KEY,
        'anthropic-version': '2023-06-01'
      }
    };
    const proxyReq = https.request(options, proxyRes => {
      res.writeHead(proxyRes.statusCode, { 'Content-Type': 'application/json' });
      proxyRes.pipe(res);
    });
    proxyReq.on('error', err => {
      res.writeHead(502);
      res.end(JSON.stringify({ error: { message: 'Proxy error: ' + err.message } }));
    });
    proxyReq.write(body);
    proxyReq.end();
    return;
  }

  // Real Python execution
  if (req.method === 'POST' && req.url === '/api/run') {
    let body;
    try { body = JSON.parse(await readBody(req)); }
    catch { res.writeHead(400); res.end(JSON.stringify({ error: 'Invalid JSON' })); return; }

    const code = body.code || '';
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'pyrunner-'));
    const scriptPath = path.join(tmpDir, 'script.py');

    // Wrap code: set non-interactive backend, intercept plt.show() to save figures
    const wrapped = `
import sys, os
os.chdir(${JSON.stringify(tmpDir)})
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as _plt
    _plot_count = [0]
    _orig_show = _plt.show
    def _patched_show(*a, **kw):
        _plt.savefig(os.path.join(${JSON.stringify(tmpDir)}, f'plot_{_plot_count[0]}.png'), bbox_inches='tight')
        _plot_count[0] += 1
        _plt.close('all')
    _plt.show = _patched_show
except ImportError:
    pass

${code}
`;
    fs.writeFileSync(scriptPath, wrapped);

    execFile(PYTHON, [scriptPath], { timeout: 30000, maxBuffer: 1024 * 1024 }, (err, stdout, stderr) => {
      // Collect any saved plot images as base64
      const plots = [];
      try {
        fs.readdirSync(tmpDir)
          .filter(f => f.startsWith('plot_') && f.endsWith('.png'))
          .sort()
          .forEach(f => {
            const data = fs.readFileSync(path.join(tmpDir, f));
            plots.push('data:image/png;base64,' + data.toString('base64'));
          });
      } catch {}

      // Clean up temp dir
      try { fs.rmSync(tmpDir, { recursive: true, force: true }); } catch {}

      const exitCode = err?.code ?? 0;
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        stdout: stdout || '',
        stderr: stderr || '',
        exitCode,
        plots
      }));
    });
    return;
  }

  res.writeHead(404);
  res.end('Not found');
});

server.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
