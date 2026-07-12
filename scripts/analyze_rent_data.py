import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

df = pd.read_csv('data/immobiliare_milano_combined_600.csv')
out = 'analysis_output'
os.makedirs(out, exist_ok=True)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

print("=" * 60)
print("MILANO ADVERTISED-RENT SNAPSHOT - {} LISTINGS".format(len(df)))
print("=" * 60)

# 1. Price Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df['prezzo'], bins=50, color='#3498db', edgecolor='white', alpha=0.8)
axes[0].set_title('Price Distribution (All)', fontweight='bold')
axes[0].set_xlabel('Monthly Rent (EUR)')
axes[0].set_ylabel('Count')
axes[0].axvline(df['prezzo'].median(), color='red', linestyle='--', label='Median: EUR {:.0f}'.format(df.prezzo.median()))
axes[0].legend()

q99 = df['prezzo'].quantile(0.99)
df_clean = df[df['prezzo'] <= q99]
axes[1].hist(df_clean['prezzo'], bins=40, color='#2ecc71', edgecolor='white', alpha=0.8)
axes[1].set_title('Zoomed view below 99th pct (EUR {:.0f})'.format(q99), fontweight='bold')
axes[1].set_xlabel('Monthly Rent (EUR)')
axes[1].axvline(df_clean['prezzo'].median(), color='red', linestyle='--', label='Median: EUR {:.0f}'.format(df_clean.prezzo.median()))
axes[1].legend()
plt.tight_layout()
plt.savefig('{}/01_price_distribution.png'.format(out))
plt.close()
print("\n1. Price Distribution - Median: EUR {:.0f}, Mean: EUR {:.0f}".format(df.prezzo.median(), df.prezzo.mean()))

# 2. Top 15 Neighborhoods
top_neigh = df['quartiere'].value_counts().head(15)
fig, ax = plt.subplots(figsize=(10, 6))
top_neigh.plot(kind='barh', color='#9b59b6', edgecolor='white', ax=ax)
ax.set_title('Top 15 Neighborhoods by Listing Count', fontweight='bold')
ax.set_xlabel('Number of Listings')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('{}/02_top_neighborhoods.png'.format(out))
plt.close()
print("\n2. Top Neighborhoods: {}".format(', '.join(top_neigh.head(5).index.tolist())))

# 3. Price by Top Neighborhoods (boxplot)
top10 = df['quartiere'].value_counts().head(10).index
df_top10 = df[df['quartiere'].isin(top10)]
fig, ax = plt.subplots(figsize=(12, 6))
order = df_top10.groupby('quartiere')['prezzo'].median().sort_values(ascending=False).index
sns.boxplot(data=df_top10, y='quartiere', x='prezzo', order=order, ax=ax, palette='viridis')
ax.set_title('Price Distribution by Top 10 Neighborhoods', fontweight='bold')
ax.set_xlabel('Monthly Rent (EUR)')
ax.set_ylabel('')
plt.tight_layout()
plt.savefig('{}/03_price_by_neighborhood.png'.format(out))
plt.close()
print("\n3. Most Expensive Neighborhoods (median):")
for n in order[:5]:
    med = df_top10[df_top10.quartiere == n].prezzo.median()
    print("   {}: EUR {:.0f}".format(n, med))

# 4. Area vs Price
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(df['superficie'], df['prezzo'], c=df['stanze'], cmap='coolwarm', alpha=0.5, s=20, edgecolors='white', linewidth=0.3)
plt.colorbar(scatter, label='Rooms')
ax.set_title('Area vs Price (colored by rooms)', fontweight='bold')
ax.set_xlabel('Area (m2)')
ax.set_ylabel('Monthly Rent (EUR)')
area_price = df[['superficie', 'prezzo']].dropna()
area_price = area_price[(area_price['superficie'] > 0) & (area_price['prezzo'] > 0)]
z = np.polyfit(area_price['superficie'], area_price['prezzo'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['superficie'].min(), df['superficie'].max(), 100)
ax.plot(x_line, p(x_line), 'r--', alpha=0.8, label='Trend: EUR {:.0f}/m2'.format(z[0]))
ax.legend()
plt.tight_layout()
plt.savefig('{}/04_area_vs_price.png'.format(out))
plt.close()
print("\n4. Descriptive linear slope: ~EUR {:.0f} more monthly rent per additional m2 (not causal)".format(z[0]))

# 5. Energy Class
energy = df['classe energetica'].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#27ae60', '#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#c0392b', '#8e44ad']
energy.plot(kind='bar', color=colors[:len(energy)], edgecolor='white', ax=ax)
ax.set_title('Energy Class Distribution', fontweight='bold')
ax.set_xlabel('Energy Class')
ax.set_ylabel('Count')
plt.tight_layout()
plt.savefig('{}/05_energy_class.png'.format(out))
plt.close()
print("\n5. Energy Classes: {}".format(dict(energy)))

# 6. Price by Rooms
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
room_price = df.groupby('stanze')['prezzo'].agg(['median', 'count'])
room_price = room_price[room_price['count'] >= 5]
room_price['median'].plot(kind='bar', color='#e67e22', edgecolor='white', ax=axes[0])
axes[0].set_title('Median Price by Rooms', fontweight='bold')
axes[0].set_xlabel('Rooms')
axes[0].set_ylabel('Median Monthly Rent (EUR)')
room_price['count'].plot(kind='bar', color='#3498db', edgecolor='white', ax=axes[1])
axes[1].set_title('Listings by Room Count', fontweight='bold')
axes[1].set_xlabel('Rooms')
axes[1].set_ylabel('Count')
plt.tight_layout()
plt.savefig('{}/06_price_by_rooms.png'.format(out))
plt.close()
print("\n6. Price by Rooms: {}".format({int(k): int(v) for k, v in room_price['median'].items()}))

# 7. Amenity Prevalence
amenities = ['arredato', 'balcone', 'fibra ottica', 'impianto tv', 'esposizione esterna',
             'cancello elettrico', 'cantina', 'impianto allarme', 'piscina', 'vista mare']
amen_pct = df[amenities].eq(1).mean() * 100
fig, ax = plt.subplots(figsize=(10, 5))
amen_pct.sort_values().plot(kind='barh', color='#1abc9c', edgecolor='white', ax=ax)
ax.set_title('Amenities Explicitly Present (%)', fontweight='bold')
ax.set_xlabel('% of Listings')
for i, v in enumerate(amen_pct.sort_values()):
    ax.text(v + 0.5, i, '{:.0f}%'.format(v), va='center', fontsize=9)
plt.tight_layout()
plt.savefig('{}/07_amenities.png'.format(out))
plt.close()
print("\n7. Top Amenities:")
for a, p in amen_pct.sort_values(ascending=False).head(5).items():
    print("   {}: {:.0f}%".format(a, p))

# 8. Price per m2
df['price_per_sqm'] = np.where(df['superficie'] > 0, df['prezzo'] / df['superficie'], np.nan)
top15_ppsm = df.groupby('quartiere')['price_per_sqm'].agg(['median', 'count'])
top15_ppsm = top15_ppsm[top15_ppsm['count'] >= 3].sort_values('median', ascending=False).head(15)
fig, ax = plt.subplots(figsize=(10, 6))
top15_ppsm['median'].plot(kind='barh', color='#e74c3c', edgecolor='white', ax=ax)
ax.set_title('Price per m2 by Neighborhood (min 3 listings)', fontweight='bold')
ax.set_xlabel('Median EUR/m2')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('{}/08_price_per_sqm.png'.format(out))
plt.close()
print("\n8. Highest EUR/m2:")
for n, row in top15_ppsm.head(5).iterrows():
    print("   {}: EUR {:.0f}/m2 ({} listings)".format(n, row['median'], int(row['count'])))

# 9. Correlation Heatmap
num_cols = ['prezzo', 'superficie', 'stanze', 'bagni', 'posti auto', 'ultimo piano']
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='RdYlBu_r', center=0, ax=ax, fmt='.2f', square=True)
ax.set_title('Feature Correlation Heatmap', fontweight='bold')
plt.tight_layout()
plt.savefig('{}/09_correlation_heatmap.png'.format(out))
plt.close()

# Summary
print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)
print("Total listings: {}".format(len(df)))
print("Price range: EUR {} - EUR {}".format(int(df.prezzo.min()), int(df.prezzo.max())))
print("Median price: EUR {:.0f}".format(df.prezzo.median()))
print("Mean price: EUR {:.0f}".format(df.prezzo.mean()))
print("Median area: {:.0f} m2".format(df.superficie.median()))
print("Median rooms: {:.0f}".format(df.stanze.median()))
print("Median bathrooms: {:.0f}".format(df.bagni.median()))
print("Neighborhoods: {}".format(df.quartiere.nunique()))
print("Furnished explicitly present: {:.0f}%".format(df.arredato.eq(1).mean() * 100))
print("Balcony explicitly present: {:.0f}%".format(df.balcone.eq(1).mean() * 100))
print("Central heating explicitly present: {:.0f}%".format(df['riscaldamento centralizzato'].eq(1).mean() * 100))
print("Energy class present: {}/{}".format(df['classe energetica'].notna().sum(), len(df)))
print("\nCharts saved to: {}/".format(out))
print("Files: 01-09 PNG charts")
