"""Génération d'un site cartographique WebGL statique."""

from __future__ import annotations

import html
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from .data import _normalize_years, get_dataframe


SEVERITY_TO_WEB_CODE = {
    "Indemne seulement": 0,
    "Léger": 1,
    "Blessé hospitalisé": 2,
    "Mortel": 3,
    "Inconnue": -1,
}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def build_web_map(
    years: Iterable[int] | int | None = None,
    communes: Iterable[str | int] | str | int | None = None,
    *,
    data_dir: str | Path = "data",
    output_dir: str | Path = "site",
    title: str = "Carte des accidents corporels",
    fallback_center: tuple[float, float] = (46.5, 2.2),
    fallback_zoom: int = 5,
    basemap_url: str = "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    basemap_attribution: str = "© OpenStreetMap contributors",
) -> Path:
    """Construit un site cartographique WebGL et retourne son fichier ``index.html``.

    Les points sont répartis dans un JSON compact par année. Seules les données
    nécessaires à la carte sont publiées : longitude, latitude, gravité, code INSEE,
    identifiant d'accident et date. Les adresses et données relatives aux usagers ne
    sont jamais écrites dans le site.
    """
    data_directory = Path(data_dir).expanduser()
    selected_years = _normalize_years(years, data_directory)
    if len(fallback_center) != 2:
        raise ValueError("`fallback_center` doit contenir latitude et longitude")
    if not -90 <= fallback_center[0] <= 90 or not -180 <= fallback_center[1] <= 180:
        raise ValueError("`fallback_center` contient des coordonnées invalides")
    if not 0 <= fallback_zoom <= 22:
        raise ValueError("`fallback_zoom` doit être compris entre 0 et 22")

    destination = Path(output_dir)
    points_dir = destination / "points"
    points_dir.mkdir(parents=True, exist_ok=True)
    manifest_years: list[dict[str, Any]] = []
    total_accidents = 0
    total_mapped = 0
    longitude_min = math.inf
    longitude_max = -math.inf
    latitude_min = math.inf
    latitude_max = -math.inf

    for year in selected_years:
        accidents = get_dataframe(
            years=year,
            communes=communes,
            data_dir=data_dir,
        )
        mapped = accidents[accidents["has_coordinates"]].copy()
        dates = mapped["date"].dt.strftime("%Y-%m-%d").fillna("")
        severity_codes = mapped["severity"].map(SEVERITY_TO_WEB_CODE).fillna(-1)
        points = [
            [
                round(float(longitude), 5),
                round(float(latitude), 5),
                int(severity),
                str(commune),
                str(accident_id),
                str(date),
            ]
            for longitude, latitude, severity, commune, accident_id, date in zip(
                mapped["longitude"],
                mapped["latitude"],
                severity_codes,
                mapped["commune_code"],
                mapped["accident_id"],
                dates,
            )
        ]
        relative_path = f"points/{year}.json"
        _write_json(points_dir / f"{year}.json", points)
        severity_counts = {
            str(code): int((severity_codes == code).sum())
            for code in (-1, 0, 1, 2, 3)
        }
        manifest_years.append(
            {
                "year": year,
                "file": relative_path,
                "accidents": len(accidents),
                "mapped": len(mapped),
                "severity": severity_counts,
            }
        )
        total_accidents += len(accidents)
        total_mapped += len(mapped)
        if not mapped.empty:
            longitude_min = min(longitude_min, float(mapped["longitude"].min()))
            longitude_max = max(longitude_max, float(mapped["longitude"].max()))
            latitude_min = min(latitude_min, float(mapped["latitude"].min()))
            latitude_max = max(latitude_max, float(mapped["latitude"].max()))

    if total_mapped:
        longitude_span = max(longitude_max - longitude_min, 1e-6)
        latitude_span = max(latitude_max - latitude_min, 1e-6)
        geographic_span = max(longitude_span, latitude_span * 1.4)
        automatic_zoom = max(1, min(16, int(math.floor(math.log2(360 / geographic_span))) - 1))
        view = {
            "longitude": (longitude_min + longitude_max) / 2,
            "latitude": (latitude_min + latitude_max) / 2,
            "zoom": automatic_zoom,
            "bounds": [[longitude_min, latitude_min], [longitude_max, latitude_max]],
        }
    else:
        view = {
            "longitude": float(fallback_center[1]),
            "latitude": float(fallback_center[0]),
            "zoom": int(fallback_zoom),
            "bounds": None,
        }

    manifest = {
        "title": title,
        "years": manifest_years,
        "totalAccidents": total_accidents,
        "totalMapped": total_mapped,
        "generatedFrom": "BAAC — Ministère de l'Intérieur / ONISR",
        "view": view,
    }
    _write_json(destination / "manifest.json", manifest)
    (destination / ".nojekyll").write_text("", encoding="utf-8")
    (destination / "index.html").write_text(
        _html_template(title, basemap_url, basemap_attribution), encoding="utf-8"
    )
    return destination / "index.html"


def _html_template(title: str, basemap_url: str, basemap_attribution: str) -> str:
    safe_title = json.dumps(title, ensure_ascii=False)
    safe_basemap_url = json.dumps(basemap_url, ensure_ascii=False)
    escaped_title = html.escape(title)
    escaped_attribution = html.escape(basemap_attribution)
    return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="description" content="Carte interactive des accidents corporels BAAC">
  <title>{escaped_title}</title>
  <script src="https://unpkg.com/deck.gl@9.1.14/dist.min.js"></script>
  <style>
    :root {{ color-scheme: light; font-family: Inter, system-ui, sans-serif; }}
    * {{ box-sizing: border-box; }}
    html, body, #map {{ width: 100%; height: 100%; margin: 0; }}
    body {{ overflow: hidden; background: #e9eef1; color: #17324d; }}
    #panel {{
      position: absolute; z-index: 5; top: 16px; left: 16px; width: min(370px, calc(100% - 32px));
      max-height: calc(100% - 32px); overflow: auto; padding: 17px 18px;
      background: rgba(255,255,255,.96); border: 1px solid #c9d2d8; border-radius: 9px;
      box-shadow: 0 4px 20px rgba(23,50,77,.18);
    }}
    h1 {{ font-size: 18px; line-height: 1.25; margin: 0 0 7px; }}
    .meta, .footnote {{ color: #596b78; font-size: 12px; line-height: 1.45; }}
    #progress {{ margin: 10px 0; font-size: 13px; font-weight: 650; }}
    progress {{ width: 100%; height: 8px; accent-color: #315b78; }}
    fieldset {{ border: 0; border-top: 1px solid #dbe1e5; margin: 12px 0 0; padding: 11px 0 0; }}
    legend {{ padding: 0; font-size: 13px; font-weight: 700; }}
    label {{ font-size: 13px; cursor: pointer; }}
    .severity {{ display: grid; grid-template-columns: 1fr 1fr; gap: 7px; }}
    .dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin:0 5px; }}
    .years {{ display:grid; grid-template-columns:repeat(4,1fr); gap:5px; margin-top:7px; }}
    .actions {{ display:flex; gap:8px; margin-top:8px; }}
    button {{ border:1px solid #9eacb5; background:#f7f9fa; border-radius:5px; padding:5px 8px; cursor:pointer; }}
    #tooltip {{
      position:absolute; z-index:6; display:none; pointer-events:none; max-width:280px;
      background:#172b3a; color:white; padding:8px 10px; border-radius:5px;
      font-size:12px; line-height:1.45;
    }}
    @media (max-width: 620px) {{ #panel {{ top:8px; left:8px; width:calc(100% - 16px); max-height:48%; }} }}
  </style>
</head>
<body>
  <div id="map"></div>
  <aside id="panel">
    <h1 id="title">{escaped_title}</h1>
    <div class="meta" id="summary">Préparation de la carte…</div>
    <div id="progress">Chargement du manifeste…</div>
    <progress id="progressbar" value="0" max="1"></progress>
    <fieldset>
      <legend>Gravité</legend>
      <div class="severity">
        <label><input type="checkbox" data-severity="3" checked><span class="dot" style="background:#d00000"></span>Mortel</label>
        <label><input type="checkbox" data-severity="2" checked><span class="dot" style="background:#f28c00"></span>Hospitalisé</label>
        <label><input type="checkbox" data-severity="1" checked><span class="dot" style="background:#ffd800"></span>Léger</label>
        <label><input type="checkbox" data-severity="0" checked><span class="dot" style="background:#6baed6"></span>Indemne</label>
        <label><input type="checkbox" data-severity="-1" checked><span class="dot" style="background:#888"></span>Inconnue</label>
      </div>
    </fieldset>
    <fieldset>
      <legend>Années</legend>
      <div class="actions"><button id="allYears">Toutes</button><button id="noYears">Aucune</button></div>
      <div class="years" id="years"></div>
    </fieldset>
    <p class="footnote">Données : BAAC — Ministère de l’Intérieur / ONISR,
      Licence Ouverte. Fond : {escaped_attribution}. Seuls les accidents
      disposant de coordonnées valides sont positionnés ; les adresses et
      caractéristiques individuelles ne sont pas publiées.</p>
  </aside>
  <div id="tooltip"></div>
  <script>
    const PAGE_TITLE = {safe_title};
    const COLORS = {{'-1':[140,140,140,150],0:[107,174,214,150],1:[255,216,0,175],2:[242,140,0,210],3:[208,0,0,235]}};
    const LABELS = {{'-1':'Inconnue',0:'Indemne seulement',1:'Léger',2:'Blessé hospitalisé',3:'Mortel'}};
    const loaded = new Map();
    const selectedYears = new Set();
    const selectedSeverities = new Set([-1,0,1,2,3]);
    let manifest;
    let deckInstance;

    function basemapLayer() {{
      return new deck.TileLayer({{
        id:'basemap', data:{safe_basemap_url},
        minZoom:0, maxZoom:19, tileSize:256,
        renderSubLayers: props => {{
          const bounds = props.tile.boundingBox;
          return new deck.BitmapLayer(props, {{
            data:null,
            image:props.data,
            bounds:[bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]]
          }});
        }}
      }});
    }}

    function layers() {{
      const result = [basemapLayer()];
      for (const item of manifest.years) {{
        const points = loaded.get(item.year);
        if (!points) continue;
        result.push(new deck.ScatterplotLayer({{
          id:`accidents-${{item.year}}`, data:points, visible:selectedYears.has(item.year),
          pickable:true, opacity:0.85, stroked:false, radiusUnits:'pixels', radiusMinPixels:2,
          getPosition:d=>[d[0],d[1]],
          getRadius:d=>selectedSeverities.has(d[2]) ? (d[2]===3?6:d[2]===2?4:2.5) : 0,
          getFillColor:d=>COLORS[d[2]] || COLORS[-1],
          updateTriggers:{{getRadius:[...selectedSeverities].sort().join(',')}}
        }}));
      }}
      return result;
    }}

    function redraw() {{ if (deckInstance) deckInstance.setProps({{layers:layers()}}); updateVisibleCount(); }}
    function updateVisibleCount() {{
      if (!manifest) return;
      let count=0;
      for (const item of manifest.years) if(selectedYears.has(item.year)) {{
        for (const code of selectedSeverities) count += item.severity[String(code)] || 0;
      }}
      document.getElementById('summary').textContent = `${{count.toLocaleString('fr-FR')}} points affichés — ${{manifest.totalMapped.toLocaleString('fr-FR')}} géolocalisés sur ${{manifest.totalAccidents.toLocaleString('fr-FR')}} accidents`;
    }}

    async function start() {{
      manifest = await fetch('manifest.json').then(response => response.json());
      document.getElementById('title').textContent = manifest.title || PAGE_TITLE;
      const yearsBox=document.getElementById('years');
      for(const item of manifest.years) {{
        selectedYears.add(item.year);
        const label=document.createElement('label');
        label.innerHTML=`<input type="checkbox" data-year="${{item.year}}" checked> ${{item.year}}`;
        yearsBox.appendChild(label);
      }}
      deckInstance = new deck.Deck({{
        parent:document.getElementById('map'), initialViewState:manifest.view,
        controller:true, layers:layers(),
        onHover:({{object,x,y}})=>{{
          const tooltip=document.getElementById('tooltip');
          if(!object){{tooltip.style.display='none';return;}}
          tooltip.style.display='block'; tooltip.style.left=`${{x+12}}px`; tooltip.style.top=`${{y+12}}px`;
          tooltip.innerHTML=`<strong>${{LABELS[object[2]]}}</strong><br>Date : ${{object[5] || 'inconnue'}}<br>Commune INSEE : ${{object[3]}}<br>Accident : ${{object[4]}}`;
        }}
      }});
      bindControls();
      const bar=document.getElementById('progressbar'); bar.max=manifest.years.length;
      for(let index=0; index<manifest.years.length; index++) {{
        const item=manifest.years[index];
        document.getElementById('progress').textContent=`Chargement ${{item.year}} — ${{index+1}}/${{manifest.years.length}}`;
        const points=await fetch(item.file).then(response=>response.json());
        loaded.set(item.year,points); bar.value=index+1; redraw();
      }}
      document.getElementById('progress').textContent='Toutes les années sont chargées.';
    }}

    function bindControls() {{
      document.querySelectorAll('[data-severity]').forEach(input=>input.addEventListener('change',()=>{{
        const code=Number(input.dataset.severity); input.checked?selectedSeverities.add(code):selectedSeverities.delete(code); redraw();
      }}));
      document.querySelectorAll('[data-year]').forEach(input=>input.addEventListener('change',()=>{{
        const year=Number(input.dataset.year); input.checked?selectedYears.add(year):selectedYears.delete(year); redraw();
      }}));
      document.getElementById('allYears').onclick=()=>setAllYears(true);
      document.getElementById('noYears').onclick=()=>setAllYears(false);
    }}
    function setAllYears(checked) {{
      selectedYears.clear();
      document.querySelectorAll('[data-year]').forEach(input=>{{input.checked=checked;if(checked)selectedYears.add(Number(input.dataset.year));}});
      redraw();
    }}
    start().catch(error=>{{document.getElementById('progress').textContent=`Erreur : ${{error.message}}`;console.error(error);}});
  </script>
</body>
</html>
"""
