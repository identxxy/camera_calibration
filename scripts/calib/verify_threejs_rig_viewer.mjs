import fs from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright-core");

const [url, outputDir] = process.argv.slice(2);

if (!url || !outputDir) {
  console.error("Usage: node verify_threejs_rig_viewer.mjs <url> <output-dir>");
  process.exit(2);
}

async function canvasStats(page) {
  return await page.evaluate(() => {
    const canvas = document.querySelector("canvas");
    if (!canvas) {
      return {error: "missing canvas"};
    }
    const gl = canvas.getContext("webgl2") || canvas.getContext("webgl") || canvas.getContext("experimental-webgl");
    if (!gl) {
      return {error: "missing webgl context", width: canvas.width, height: canvas.height};
    }
    const width = canvas.width;
    const height = canvas.height;
    const pixels = new Uint8Array(width * height * 4);
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    let sampled = 0;
    let nonBackground = 0;
    let bright = 0;
    const stride = 8 * 4;
    for (let i = 0; i < pixels.length; i += stride) {
      const r = pixels[i];
      const g = pixels[i + 1];
      const b = pixels[i + 2];
      const contrast = Math.abs(r - 20) + Math.abs(g - 20) + Math.abs(b - 20);
      sampled += 1;
      if (contrast > 18) {
        nonBackground += 1;
      }
      if (r + g + b > 160) {
        bright += 1;
      }
    }
    const viewer = window.__rigViewer || {};
    return {
      width,
      height,
      sampled,
      nonBackground,
      bright,
      selectedIndex: viewer.getSelectedIndex ? viewer.getSelectedIndex() : null,
      cameraPosition: viewer.camera ? viewer.camera.position.toArray() : null,
      frustumState: viewer.getFrustumState ? viewer.getFrustumState() : null,
    };
  });
}

function distance(a, b) {
  if (!a || !b || a.length !== b.length) {
    return 0;
  }
  let sum = 0;
  for (let i = 0; i < a.length; ++i) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

async function checkViewport(browser, viewport, name) {
  const page = await browser.newPage({viewport});
  await page.goto(url, {waitUntil: "networkidle"});
  await page.waitForFunction(() => window.__rigViewer && document.querySelector("canvas"));
  await page.waitForTimeout(800);

  const before = await canvasStats(page);
  const screenshotPath = path.join(outputDir, `verify_${name}.png`);
  await page.screenshot({path: screenshotPath});

  await page.evaluate(() => {
    window.__rigViewer.setOrientationForTest(0.34, -0.22, 0.61);
    const near = document.getElementById("near-slider");
    const far = document.getElementById("far-slider");
    near.value = "0.25";
    far.value = "1.20";
    near.dispatchEvent(new Event("input", {bubbles: true}));
    far.dispatchEvent(new Event("input", {bubbles: true}));
  });
  const tiltedControls = await canvasStats(page);
  await page.click("#reset-up");
  await page.click("#toggle-overlap");
  await page.waitForTimeout(350);
  const afterControls = await canvasStats(page);
  const controlsScreenshotPath = path.join(outputDir, `verify_${name}_controls.png`);
  await page.screenshot({path: controlsScreenshotPath});

  const start = before.cameraPosition;
  await page.mouse.move(Math.floor(viewport.width * 0.46), Math.floor(viewport.height * 0.45));
  await page.mouse.down();
  await page.mouse.move(Math.floor(viewport.width * 0.68), Math.floor(viewport.height * 0.58), {steps: 10});
  await page.mouse.up();
  await page.waitForTimeout(350);
  const after = await canvasStats(page);
  await page.close();

  const movement = distance(start, after.cameraPosition);
  const minNonBackground = Math.max(80, Math.floor(before.sampled * 0.003));
  const state = afterControls.frustumState || {};
  const tiltedState = tiltedControls.frustumState || {};
  const ok = !before.error
    && before.nonBackground >= minNonBackground
    && movement > 1e-4
    && state.frustumMeshCount >= 1
    && state.overlapVisible === true
    && Math.abs(state.near - 0.25) < 1e-6
    && Math.abs(state.far - 1.20) < 1e-6
    && state.overlapMeshCount >= 1
    && Array.isArray(state.up)
    && Array.isArray(state.frameQuaternion)
    && tiltedState.up[1] < 0.99
    && Math.abs(state.up[0]) < 1e-5
    && Math.abs(state.up[1] - 1) < 1e-5
    && Math.abs(state.up[2]) < 1e-5;
  return {
    name,
    viewport,
    screenshotPath,
    controlsScreenshotPath,
    before,
    tiltedControls,
    afterControls,
    after,
    cameraMovement: movement,
    ok,
  };
}

await fs.mkdir(outputDir, {recursive: true});

const browser = await chromium.launch({
  headless: true,
  executablePath: "/usr/bin/google-chrome",
  args: [
    "--no-sandbox",
    "--enable-webgl",
    "--ignore-gpu-blocklist",
    "--use-gl=swiftshader",
  ],
});

try {
  const results = [];
  results.push(await checkViewport(browser, {width: 1280, height: 820}, "desktop"));
  results.push(await checkViewport(browser, {width: 390, height: 844, isMobile: true}, "mobile"));
  const summaryPath = path.join(outputDir, "verify_summary.json");
  await fs.writeFile(summaryPath, JSON.stringify({url, results}, null, 2));
  const failed = results.filter((result) => !result.ok);
  console.log(JSON.stringify({summaryPath, results}, null, 2));
  if (failed.length) {
    process.exit(1);
  }
} finally {
  await browser.close();
}
