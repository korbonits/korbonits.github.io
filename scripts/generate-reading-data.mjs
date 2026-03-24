// Generates public/reading-data.json from Goodreads CSV export
// Usage: node scripts/generate-reading-data.mjs <path/to/goodreads_library_export.csv>
import { parse } from 'csv-parse/sync';
import { readFileSync, writeFileSync } from 'fs';

const CSV = process.argv[2];
if (!CSV) { console.error('Usage: node scripts/generate-reading-data.mjs <path/to/goodreads_library_export.csv>'); process.exit(1); }

const rows = parse(readFileSync(CSV), { columns: true });
const read = rows.filter(r => r['Exclusive Shelf'] === 'read' && r['Date Read']);

// Books per year
const byYear = {};
for (const r of read) {
  const y = r['Date Read'].slice(0, 4);
  if (!byYear[y]) byYear[y] = { count: 0, pages: 0, rated: 0, ratingSum: 0 };
  byYear[y].count++;
  const pages = parseInt(r['Number of Pages']) || 0;
  byYear[y].pages += pages;
  const rating = parseInt(r['My Rating']) || 0;
  if (rating > 0) { byYear[y].rated++; byYear[y].ratingSum += rating; }
}

// Ratings distribution (only rated books)
const ratingDist = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 };
for (const r of read) {
  const rating = parseInt(r['My Rating']) || 0;
  if (rating > 0) ratingDist[rating]++;
}

// Top authors (by books read, min 2)
const authorCount = {};
for (const r of read) {
  const a = r['Author'];
  authorCount[a] = (authorCount[a] || 0) + 1;
}
const topAuthors = Object.entries(authorCount)
  .filter(([, n]) => n >= 2)
  .sort((a, b) => b[1] - a[1])
  .slice(0, 15)
  .map(([author, count]) => ({ author, count }));

// Summary stats
const totalPages = read.reduce((s, r) => s + (parseInt(r['Number of Pages']) || 0), 0);
const rated = read.filter(r => parseInt(r['My Rating']) > 0);
const avgRating = rated.length ? (rated.reduce((s, r) => s + parseInt(r['My Rating']), 0) / rated.length).toFixed(2) : null;

const years = Object.keys(byYear).sort();
const yearStats = years.map(y => ({ year: y, ...byYear[y], avgRating: byYear[y].rated ? (byYear[y].ratingSum / byYear[y].rated).toFixed(2) : null }));

writeFileSync('public/reading-data.json', JSON.stringify({ yearStats, ratingDist, topAuthors, summary: { total: read.length, totalPages, avgRating } }, null, 2));
console.log(`Written public/reading-data.json (${read.length} books, ${years.length} years)`);
