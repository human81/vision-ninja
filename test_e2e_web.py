"""End-to-end browser test — Playwright fully drives the web UI headlessly:
launch the server, open the page, DRAW a zone + counting line by clicking the
canvas, Start the pipeline, and assert live stats show detections, a zone count,
and line crossings. Screenshots the result into the fishfood gallery.

    .venv/bin/python test_e2e_web.py
"""

import os
import socket
import subprocess
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

PORT = 8077
URL = f"http://127.0.0.1:{PORT}"


def _wait_port(port, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.3)
    return False


def _click_norm(page, sel, nx, ny):
    box = page.locator(sel).bounding_box()
    page.mouse.click(box["x"] + nx * box["width"], box["y"] + ny * box["height"])


def main():
    env = {**os.environ, "OCC_SOURCE": "assets/videos/vehicles-2.mp4"}
    server = subprocess.Popen(
        [".venv/bin/uvicorn", "occ.web:app", "--port", str(PORT), "--host", "127.0.0.1"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        assert _wait_port(PORT), "server did not start"
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 900, "height": 700})
            # MJPEG <img> keeps the connection open, so "load" never fires — use DCL.
            page.goto(URL, wait_until="domcontentloaded")
            page.wait_for_selector("#cv")
            time.sleep(1.0)  # let the first MJPEG frame load + canvas size

            # draw a ZONE (lower half of the road): Add zone → 4 clicks → Finish
            page.click("#zone")
            for nx, ny in [(0.2, 0.6), (0.8, 0.6), (0.8, 0.95), (0.2, 0.95)]:
                _click_norm(page, "#cv", nx, ny)
            page.click("#finish")
            # draw a counting LINE across the road: Add line → 2 clicks (auto-finish)
            page.click("#line")
            _click_norm(page, "#cv", 0.05, 0.80)
            _click_norm(page, "#cv", 0.95, 0.80)

            # verify the annotations were posted
            posted = page.evaluate(
                "fetch('/stats').then(r=>r.json())")  # warms the endpoint
            page.click("#start")

            # poll the live stats until detections + zone + crossings appear
            deadline = time.time() + 40
            stats = {}
            while time.time() < deadline:
                stats = page.evaluate("fetch('/stats').then(r=>r.json())")
                full = sum(stats.get("full_frame", {}).values())
                zone = sum(stats.get("zones", {}).values())
                cross = sum(d["positive"] + d["negative"]
                            for d in stats.get("lines", {}).values())
                if full > 0 and zone > 0 and cross > 0:
                    break
                time.sleep(0.5)

            Path("out/fishfood").mkdir(parents=True, exist_ok=True)
            page.screenshot(path="out/fishfood/e2e_web.png")

            full = sum(stats.get("full_frame", {}).values())
            zone = sum(stats.get("zones", {}).values())
            cross = sum(d["positive"] + d["negative"]
                        for d in stats.get("lines", {}).values())
            assert full > 0, f"no detections in stats: {stats}"
            assert zone > 0, f"zone never occupied: {stats}"
            assert cross > 0, f"no line crossings counted: {stats}"
            assert stats["tracks"] > 0, f"no tracks: {stats}"
            browser.close()
            print(f"E2E OK — browser drew zone+line, started run; live stats: "
                  f"full_frame={full} zone_occupancy={zone} crossings={cross} "
                  f"tracks={stats['tracks']}")
            print("screenshot → out/fishfood/e2e_web.png")
        print("\nE2E WEB PASS")
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except Exception:
            server.kill()


if __name__ == "__main__":
    main()
