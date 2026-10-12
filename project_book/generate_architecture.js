const path = require('path');
const { chromium } = require('C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1000, height: 430 }, deviceScaleFactor: 2 });
    await page.goto('file:///' + path.join(__dirname, 'architecture.svg').replace(/\\/g, '/'));
    await page.evaluate(() => document.fonts.ready);
    await page.screenshot({ path: path.join(__dirname, 'architecture.png') });
  } finally {
    await browser.close();
  }
})();
