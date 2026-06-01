#!/usr/bin/env node
/**
 * Convert each HTML slide to PDF page and merge
 */
const { chromium } = require('playwright');
const path = require('path');

(async () => {
  const htmlPath = path.resolve(__dirname, 'slides-v2.html');
  const pdfPath = path.resolve(__dirname, 'slides-v2.pdf');
  const fileUrl = `file://${htmlPath}`;

  console.log('Launching Chrome...');
  const browser = await chromium.launch({
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  console.log('Loading HTML...');
  await page.goto(fileUrl, { waitUntil: 'networkidle', timeout: 30000 });
  await page.waitForTimeout(3000);

  // Hide navigation bar before PDF generation
  await page.evaluate(() => {
    const nav = document.querySelector('.nav');
    if (nav) nav.style.display = 'none';
  });

  // Get total slide count
  const slideCount = await page.evaluate(() => {
    return document.querySelectorAll('.slide').length;
  });
  console.log(`Found ${slideCount} slides`);

  const pdfBuffers = [];

  for (let i = 0; i < slideCount; i++) {
    // Activate slide i
    await page.evaluate((idx) => {
      document.querySelectorAll('.slide').forEach((s, j) => {
        s.classList.toggle('active', j === idx);
      });
    }, i);

    // Wait for mermaid to render
    await page.waitForTimeout(1000);

    const slidePdf = await page.pdf({
      width: '1920px',
      height: '1080px',
      printBackground: true,
      margin: { top: '0', right: '0', bottom: '0', left: '0' }
    });

    pdfBuffers.push(slidePdf);
    process.stdout.write(`\rSlide ${i + 1}/${slideCount} done`);
  }

  console.log('\nMerging PDFs...');

  // Use pdf-lib to merge
  const { PDFDocument } = require('pdf-lib');
  const mergedPdf = await PDFDocument.create();

  for (const pdfBytes of pdfBuffers) {
    const srcDoc = await PDFDocument.load(pdfBytes);
    const copiedPages = await mergedPdf.copyPages(srcDoc, srcDoc.getPageIndices());
    copiedPages.forEach(page => mergedPdf.addPage(page));
  }

  const finalBytes = await mergedPdf.save();
  require('fs').writeFileSync(pdfPath, finalBytes);

  await browser.close();
  console.log(`Done: ${pdfPath} (${slideCount} pages)`);
})();