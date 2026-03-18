// Generates public/assets/reading-by-year.svg from Goodreads CSV export
import { parse } from 'csv-parse/sync';
import { readFileSync, writeFileSync } from 'fs';

const CSV = process.argv[2];
if (!CSV) { console.error('Usage: node scripts/generate-reading-chart.mjs <path/to/goodreads_export.csv>'); process.exit(1); }

const rows = parse(readFileSync(CSV), { columns: true });
const read = rows.filter(r => r['Exclusive Shelf'] === 'read' && r['Date Read']);

const byYear = {};
for (const r of read) {
  const y = r['Date Read'].slice(0, 4);
  byYear[y] = (byYear[y] || 0) + 1;
}

// Only show 2016+ (meaningful data)
const years = Object.keys(byYear).filter(y => parseInt(y) >= 2016).sort();
const counts = years.map(y => byYear[y]);
const max = Math.max(...counts);

const W = 640, H = 320, PAD = { top: 20, right: 20, bottom: 40, left: 40 };
const chartW = W - PAD.left - PAD.right;
const chartH = H - PAD.top - PAD.bottom;
const barW = Math.floor(chartW / years.length) - 4;

const bars = years.map((y, i) => {
  const bh = Math.round((counts[i] / max) * chartH);
  const x = PAD.left + i * (chartW / years.length) + 2;
  const yPos = PAD.top + chartH - bh;
  const color = (y === '2020' || y === '2021' || y === '2022') ? '#4f8ef7' : '#94a3b8';
  return `
  <rect x="${x}" y="${yPos}" width="${barW}" height="${bh}" fill="${color}" rx="2"/>
  <text x="${x + barW/2}" y="${yPos - 4}" text-anchor="middle" font-size="10" fill="#94a3b8">${counts[i]}</text>
  <text x="${x + barW/2}" y="${H - PAD.bottom + 14}" text-anchor="middle" font-size="10" fill="#94a3b8">${y}</text>`;
}).join('');

const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" style="width:100%;max-width:${W}px;font-family:sans-serif;background:transparent">
  <!-- y-axis -->
  <line x1="${PAD.left}" y1="${PAD.top}" x2="${PAD.left}" y2="${PAD.top + chartH}" stroke="#334155" stroke-width="1"/>
  <!-- x-axis -->
  <line x1="${PAD.left}" y1="${PAD.top + chartH}" x2="${W - PAD.right}" y2="${PAD.top + chartH}" stroke="#334155" stroke-width="1"/>
  ${bars}
</svg>`;

writeFileSync('public/assets/reading-by-year.svg', svg);
console.log(`Written public/assets/reading-by-year.svg (${years.length} years, max ${max} books)`);
