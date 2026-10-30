#!/usr/bin/env python3
"""E2E test for https://turbolego.github.io/abideWCAG/ — loads the model, runs a check,
and captures console/page errors to debug why 'Running inference…' never finishes.

Usage:
  python3 tests/e2e_abidewcag_pages.py            # test the live GitHub Pages site
  python3 tests/e2e_abidewcag_pages.py --local    # test the local copy via http.server
"""
import argparse, socketserver, threading, time, http.server, functools, sys
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".mjs": "text/javascript", ".js": "text/javascript",
                      ".wasm": "application/wasm", ".onnx": "application/octet-stream",
                      ".json": "application/json", ".data": "application/octet-stream"}
    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()
    def log_message(self, *a): pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true", help="serve ROOT over localhost instead of hitting Pages")
    ap.add_argument("--url", default="https://turbolego.github.io/abideWCAG/", help="base URL to test")
    ap.add_argument("--timeout", type=int, default=180, help="max seconds to wait for model load + inference")
    args = ap.parse_args()

    srv = None
    base = args.url
    if args.local:
        socketserver.ThreadingTCPServer.allow_reuse_address = True
        srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(ROOT)))
        port = srv.server_address[1]
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{port}/"

    print(f"Testing: {base}")
    failures = []

    with sync_playwright() as p:
        b = p.chromium.launch(args=["--no-sandbox"])
        ctx = b.new_context()
        pg = ctx.new_page()
        logs = []
        pg.on("console", lambda m: logs.append(f"[{m.type}] {m.text}"[:500]))
        pg.on("pageerror", lambda e: logs.append(f"[pageerror] {str(e)[:500]}"))
        pg.on("requestfailed", lambda r: logs.append(f"[reqfail] {r.url[-150:]} {r.failure}"))

        print("Navigating…")
        pg.goto(base, wait_until="domcontentloaded", timeout=30000)
        pg.wait_for_load_state("load", timeout=30000)

        # Wait for the module script to initialize (buttons exist)
        pg.wait_for_selector("#loadBtn", timeout=15000)
        print("Page loaded. H1:", pg.evaluate("document.querySelector('h1')?.textContent"))

        # --- Step 1: Load model ---
        print("\n=== STEP 1: Load model ===")
        pg.click("#loadBtn")
        deadline = time.time() + args.timeout
        loaded = False
        while time.time() < deadline:
            status = pg.evaluate("document.getElementById('status')?.textContent || ''")
            run_disabled = pg.evaluate("document.getElementById('runBtn')?.disabled")
            if run_disabled is False:
                print(f"MODEL LOADED: {status}")
                loaded = True
                break
            if "failed" in status.lower() or "error" in status.lower():
                print(f"LOAD FAILED: {status}")
                failures.append(f"model load: {status}")
                break
            if int(time.time()) % 10 == 0:
                print(f"  … {status}")
            time.sleep(2)
        if not loaded and not failures:
            failures.append("model did not load within timeout")
            last = pg.evaluate("document.getElementById('status')?.textContent")
            print(f"TIMEOUT waiting for model. Last status: {last}")

        if loaded:
            # --- Step 2: Select example and run check ---
            print("\n=== STEP 2: Run check ===")
            pg.select_option("#exampleSel", "Form input without label")
            pg.evaluate("document.getElementById('exampleSel').dispatchEvent(new Event('change'))")
            html_val = pg.evaluate("document.getElementById('htmlInput').value")
            print(f"Input HTML: {html_val[:80]}…")

            pg.click("#runBtn")
            print("Clicked run. Polling for results…")

            deadline = time.time() + 60
            done = False
            while time.time() < deadline:
                status = pg.evaluate("document.getElementById('status')?.textContent || ''")
                results_visible = pg.evaluate("document.getElementById('resultsCard')?.style.display")
                if results_visible == "block":
                    print(f"RESULTS APPEARED after poll. Status: {status}")
                    done = True
                    break
                if "failed" in status.lower() or "error" in status.lower():
                    print(f"RUN FAILED: {status}")
                    failures.append(f"inference: {status}")
                    break
                time.sleep(2)
            if not done and not failures:
                failures.append("inference never completed (stuck on 'Running inference…')")
                last = pg.evaluate("document.getElementById('status')?.textContent")
                print(f"STUCK. Status: {last}")

            if done:
                # Extract results
                violations = pg.evaluate("""
                    Array.from(document.querySelectorAll('#violationsTable tbody tr'))
                         .map(r => r.innerText.replace(/\\n/g, ' | '))
                """)
                timing = pg.evaluate("document.getElementById('timing')?.textContent")
                top = pg.evaluate("""
                    Array.from(document.querySelectorAll('#allTable tbody tr'))
                         .map(r => r.innerText.replace(/\\n/g, ' | ')).slice(0, 5)
                """)
                print(f"\nTiming: {timing}")
                print(f"Violations ({len(violations)}):")
                for v in violations[:10]:
                    print(f"  - {v}")
                print(f"Top criteria:")
                for t in top:
                    print(f"  - {t}")

                if not violations:
                    print("WARNING: no violations reported (may be correct for this example at threshold 0.5)")

        # --- Dump all console/page logs ---
        print("\n=== Console / page errors ===")
        for line in logs:
            print(line)
        if not logs:
            print("(none)")

        # Screenshot for debugging
        shot = ROOT / "tests" / "e2e_screenshot.png"
        pg.screenshot(path=str(shot), full_page=True)
        print(f"\nScreenshot: {shot}")

        b.close()

    if srv:
        srv.shutdown()

    print("\n" + "=" * 50)
    if failures:
        print("E2E FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("E2E PASSED: model loaded and inference completed with results.")
        sys.exit(0)


if __name__ == "__main__":
    main()
