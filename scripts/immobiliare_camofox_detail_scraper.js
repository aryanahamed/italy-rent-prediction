// Immobiliare.it detail-page scraper for Camofox/browser context.
// Run from an already-loaded Immobiliare.it search page in Camofox.
// It fetches search pages + listing detail pages using the browser's cookies/fingerprint.
// Output: CSV matching the notebooks' raw Italian schema.

(async () => {
  const EXPECTED = [
    'regione','citta','quartiere','prezzo','datetime','posti auto','bagni per stanza','bagni','stanze','ultimo piano','stato','classe energetica','vista mare','riscaldamento centralizzato','superficie','arredato','balcone','impianto tv','esposizione esterna','fibra ottica','cancello elettrico','cantina','giardino comune','giardino privato','impianto allarme','portiere','piscina','villa','intera proprieta','appartamento','attico','loft','mansarda'
  ];
  const METADATA = ['source_url', 'scraped_at'];
  const OUTPUT_COLUMNS = EXPECTED.concat(METADATA);

  const CONFIG = {
    city: 'Milano',
    region: 'Lombardia',
    baseUrl: 'https://www.immobiliare.it/affitto-case/milano/',
    maxListings: 50,
    maxSearchPages: 3,
    delayMs: 250,
  };

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const clean = (s) => (s || '').replace(/\s+/g, ' ').trim();
  const num = (s) => {
    const m = String(s || '').replace(/\./g, '').match(/\d+(?:,\d+)?/);
    return m ? Number(m[0].replace(',', '.')) : '';
  };
  // Blank means “not stated”. It must not be converted to a confirmed “No”.
  const has = (txt, words) => words.some((w) => txt.toLowerCase().includes(w.toLowerCase())) ? 1 : '';

  function parseLinksFromDoc(doc) {
    return [...new Set([...doc.querySelectorAll('a[href*="/annunci/"]')]
      .map((a) => a.href.split('?')[0])
      .filter(Boolean))];
  }

  async function getSearchLinks(page) {
    const url = page === 1 ? CONFIG.baseUrl : `${CONFIG.baseUrl}?pag=${page}`;
    const html = await fetch(url, { credentials: 'include' }).then((r) => r.text());
    return parseLinksFromDoc(new DOMParser().parseFromString(html, 'text/html'));
  }

  function between(text, start, end) {
    const i = text.indexOf(start);
    if (i < 0) return '';
    const j = end ? text.indexOf(end, i + start.length) : -1;
    return text.slice(i + start.length, j > i ? j : undefined);
  }

  function afterLabel(block, label, nextLabels) {
    const i = block.indexOf(label);
    if (i < 0) return '';
    const tail = block.slice(i + label.length);
    let end = tail.length;
    for (const n of nextLabels) {
      const j = tail.indexOf(n);
      if (j >= 0 && j < end) end = j;
    }
    return clean(tail.slice(0, end));
  }

  function energyClass(text) {
    const idx = text.indexOf('Efficienza energetica');
    const sub = idx >= 0 ? text.slice(idx, idx + 500) : text;
    const m = sub.match(/kWh\/m² anno\s*([A-G][0-9+]?|Esente)/i) || sub.match(/Classe energetica\s*([A-G][0-9+]?|Esente)/i);
    return m ? m[1].toUpperCase().charAt(0) : '';
  }

  function normDate(v) {
    const m = String(v || '').match(/(\d{2})\/(\d{2})\/(\d{4})/);
    return m ? `${m[3]}-${m[2]}-${m[1]}` : '';
  }

  function csvEscape(v) {
    if (v === null || v === undefined) return '';
    const s = String(v);
    return /[",\n\r]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  }

  async function parseDetail(url) {
    const html = await fetch(url, { credentials: 'include' }).then((r) => r.text());
    const doc = new DOMParser().parseFromString(html, 'text/html');
    const text = clean(doc.body.innerText);
    const title = clean(doc.querySelector('h1')?.innerText) || '';
    const chars = between(text, 'Caratteristiche', 'Informazioni sul prezzo') || '';
    const alt = between(chars, 'Altre caratteristiche', 'Vedi tutte le caratteristiche') || between(chars, 'Altre caratteristiche', 'Informazioni sul prezzo') || '';
    const featureText = clean(`${chars} ${alt} ${title}`);

    const labels = ['Tipologia','Contratto','Piano','Piani edificio','Ascensore','Superficie','Locali','Camere da letto','Cucina','Bagni','Arredato','Balcone','Terrazzo','Box, posti auto','Riscaldamento','Climatizzazione','Altre caratteristiche','Stato'];
    const allNext = labels.concat(['Vedi tutte le caratteristiche','Informazioni sul prezzo','Prezzo']);

    const tipologia = afterLabel(chars, 'Tipologia', allNext);
    const piano = afterLabel(chars, 'Piano', allNext);
    const superficie = afterLabel(chars, 'Superficie', allNext);
    const locali = afterLabel(chars, 'Locali', allNext);
    const bagni = afterLabel(chars, 'Bagni', allNext);
    const arredato = afterLabel(chars, 'Arredato', allNext);
    const balcone = afterLabel(chars, 'Balcone', allNext);
    const box = afterLabel(chars, 'Box, posti auto', allNext);
    const riscaldamento = afterLabel(chars, 'Riscaldamento', allNext);
    const statoRaw = afterLabel(chars, 'Stato', allNext);
    const priceMatch = text.match(/€\s*([0-9.]+)\/mese/i);

    const area = num(superficie);
    const rooms = num(locali);
    const baths = num(bagni);
    const parts = title.split(',').map((x) => clean(x)).filter(Boolean);
    const quartiere = parts.length >= 3 ? parts[parts.length - 2] : '';

    const row = {};
    EXPECTED.forEach((c) => row[c] = '');
    row['source_url'] = url;
    row['scraped_at'] = new Date().toISOString();
    row['regione'] = CONFIG.region;
    row['citta'] = CONFIG.city;
    row['quartiere'] = quartiere;
    row['prezzo'] = priceMatch ? Number(priceMatch[1].replace(/\./g, '')) : '';
    row['datetime'] = normDate((text.match(/Annuncio aggiornato il (\d{2}\/\d{2}\/\d{4})/) || [])[1]);
    row['posti auto'] = box ? (/no|nessun/i.test(box) ? 0 : (num(box) || 1)) : '';
    row['bagni'] = baths || '';
    row['stanze'] = rooms || '';
    row['bagni per stanza'] = (baths && rooms) ? Number((baths / rooms).toFixed(3)) : '';
    row['ultimo piano'] = piano ? (/ultimo piano/i.test(piano) ? 1 : 0) : '';
    row['stato'] = statoRaw || '';
    row['classe energetica'] = energyClass(text);
    row['vista mare'] = has(featureText, ['vista mare']);
    row['riscaldamento centralizzato'] = riscaldamento ? (/centralizzato/i.test(riscaldamento) ? 1 : 0) : '';
    row['superficie'] = area || '';
    row['arredato'] = arredato ? (/^sì|si$/i.test(arredato) ? 1 : 0) : has(featureText, ['arredato']);
    row['balcone'] = balcone ? (/^sì|si$/i.test(balcone) ? 1 : 0) : has(featureText, ['balcone']);
    row['impianto tv'] = has(featureText, ['impianto tv', 'tv satellitare']);
    row['esposizione esterna'] = has(featureText, ['esposizione esterna', 'doppia esposizione']);
    row['fibra ottica'] = has(featureText, ['fibra ottica']);
    row['cancello elettrico'] = has(featureText, ['cancello elettrico']);
    row['cantina'] = has(featureText, ['cantina']);
    row['giardino comune'] = has(featureText, ['giardino comune', 'giardino condominiale']);
    row['giardino privato'] = has(featureText, ['giardino privato']);
    row['impianto allarme'] = has(featureText, ['impianto allarme', 'allarme']);
    row['portiere'] = has(featureText, ['portineria', 'portiere']);
    row['piscina'] = has(featureText, ['piscina']);
    row['villa'] = has(`${tipologia} ${title}`, ['villa']);
    row['attico'] = has(`${tipologia} ${title}`, ['attico']);
    row['loft'] = has(`${tipologia} ${title}`, ['loft']);
    row['mansarda'] = has(`${tipologia} ${title}`, ['mansarda']);
    row['appartamento'] = has(`${tipologia} ${title}`, ['appartamento']);
    const ownershipText = `${tipologia} ${title} ${text}`;
    row['intera proprieta'] = /intera proprietà|intera proprieta/i.test(ownershipText)
      ? 1
      : (/stanza|posto letto/i.test(ownershipText) ? 0 : '');
    return row;
  }

  let links = [];
  for (let page = 1; page <= CONFIG.maxSearchPages && links.length < CONFIG.maxListings + 10; page++) {
    links = [...new Set(links.concat(await getSearchLinks(page)))];
    await sleep(CONFIG.delayMs);
  }
  links = links.slice(0, CONFIG.maxListings);

  const rows = [];
  const errors = [];
  for (const link of links) {
    try {
      rows.push(await parseDetail(link));
    } catch (err) {
      errors.push({ url: link, error: String(err) });
    }
    await sleep(CONFIG.delayMs);
  }

  const csv = [OUTPUT_COLUMNS.join(',')].concat(rows.map((r) => OUTPUT_COLUMNS.map((c) => csvEscape(r[c])).join(','))).join('\n');
  window.__immobiliareRows = rows;
  window.__immobiliareCsv = csv;
  console.log(`Scraped ${rows.length} rows. Errors: ${errors.length}`);
  console.log(csv);
  return { rows: rows.length, errors, csv };
})();
