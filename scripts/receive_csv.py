from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import urllib.parse
BASE = Path(__file__).resolve().parents[1] / 'data'
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

class H(BaseHTTPRequestHandler):
    def do_POST(self):
        name = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).get('name',['out.csv'])[0]
        name = ''.join(c for c in name if c.isalnum() or c in '._-')
        if not name or not name.lower().endswith('.csv'):
            self.send_error(400, 'A safe .csv filename is required')
            return
        n = int(self.headers.get('Content-Length','0'))
        if n <= 0 or n > MAX_UPLOAD_BYTES:
            self.send_error(413, 'Invalid upload size')
            return
        data = self.rfile.read(n)
        BASE.mkdir(parents=True, exist_ok=True)
        p = BASE/name
        try:
            with p.open('xb') as output:
                output.write(data)
        except FileExistsError:
            self.send_error(409, 'File already exists')
            return
        self.send_response(200); self.end_headers(); self.wfile.write(str(p).encode())
    def log_message(self,*a): pass
HTTPServer(('127.0.0.1',8765), H).serve_forever()
