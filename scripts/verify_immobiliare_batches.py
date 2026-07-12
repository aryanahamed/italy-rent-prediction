import pandas as pd
from pathlib import Path
expected=['regione','citta','quartiere','prezzo','datetime','posti auto','bagni per stanza','bagni','stanze','ultimo piano','stato','classe energetica','vista mare','riscaldamento centralizzato','superficie','arredato','balcone','impianto tv','esposizione esterna','fibra ottica','cancello elettrico','cantina','giardino comune','giardino privato','impianto allarme','portiere','piscina','villa','intera proprieta','appartamento','attico','loft','mansarda']
metadata=['source_url','scraped_at']
base=Path('data')
prev=[base/'immobiliare_milano_rent_enriched_100.csv',base/'immobiliare_milano_rent_enriched_101_200.csv']
new=[base/f'immobiliare_milano_rent_enriched_{s}.csv' for s in ['201_300','301_400','401_500','501_600']]
prevdf=pd.concat([pd.read_csv(p) for p in prev],ignore_index=True)
key=['source_url'] if 'source_url' in prevdf.columns else ['quartiere','prezzo','superficie','bagni','stanze']
seen_prev=set(map(tuple,prevdf[key].fillna('').astype(str).values))
seen_all=set(seen_prev)
for p in new:
    df=pd.read_csv(p)
    keys=list(map(tuple,df[key].fillna('').astype(str).values))
    overlap_prev=sum(k in seen_prev for k in keys)
    overlap_prior=sum(k in seen_all for k in keys)
    dup_rows=int(df.duplicated().sum())
    dup_keys=len(keys)-len(set(keys))
    seen_all.update(keys)
    print('\nFILE', p.resolve())
    print('shape', df.shape, 'schema_match', list(df.columns) in [expected, expected + metadata])
    print('missing_total', int(df.isna().sum().sum()))
    print('missing_by_col', {k:int(v) for k,v in df.isna().sum().items() if v})
    print('price_range', (float(df.prezzo.min()), float(df.prezzo.max())), 'surface_range', (float(df.superficie.min()), float(df.superficie.max())))
    print('neighborhoods_n', df.quartiere.nunique(dropna=True), 'sample', sorted(df.quartiere.dropna().unique())[:12])
    print('duplicate_rows', dup_rows, 'duplicate_key_rows', dup_keys, 'overlap_with_first_200_keys', overlap_prev, 'overlap_with_prior_all_keys', overlap_prior)
    print('energy_missing', int(df['classe energetica'].isna().sum()), 'stato_missing', int(df['stato'].isna().sum()))
