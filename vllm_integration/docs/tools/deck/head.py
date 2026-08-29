HEAD = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fleet MK — a static generator for fused decode megakernels</title>
<style>
  :root {
    --bg:#ffffff; --fg:#1a1a1a; --muted:#5a6472; --rule:#e2e6eb;
    --accent:#0b5fa5; --accent-soft:#eef5fb;
    --warn-bg:#fff8e6; --warn-br:#e0b400;
    --bad-bg:#fdf0ef;  --bad-br:#c0392b;
    --good-bg:#eef8f0; --good-br:#2e7d4f;
    --code-bg:#f6f7f9;
    --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  }
  * { box-sizing:border-box; }
  html,body { margin:0; background:#e9edf1; color:var(--fg);
    font:16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }

  /* ---- one slide = one 16:9 page ---- */
  .slide {
    position:relative; width:1120px; min-height:630px; margin:26px auto;
    background:var(--bg); border:1px solid #c9d2da; border-radius:6px;
    padding:44px 56px 58px; display:flex; flex-direction:column;
    box-shadow:0 2px 10px rgba(20,30,40,.09);
  }
  .slide > .num {
    position:absolute; right:22px; bottom:14px;
    font-family:var(--mono); font-size:12px; color:#9aa5b1;
  }
  .slide > .tag {
    position:absolute; left:56px; top:18px;
    font-size:11px; letter-spacing:1.4px; text-transform:uppercase;
    color:var(--accent); font-weight:700;
  }
  h1 { font-size:40px; line-height:1.15; margin:0 0 10px; letter-spacing:-0.8px; }
  h2 { font-size:29px; line-height:1.2; margin:14px 0 6px; letter-spacing:-0.4px; }
  .sub { font-size:17px; color:var(--muted); margin:0 0 18px; }
  p { margin:11px 0; }
  ul { margin:12px 0; padding-left:24px; } li { margin:9px 0; }
  b, strong { font-weight:700; }
  code { font-family:var(--mono); font-size:0.87em; background:var(--code-bg);
         padding:1px 5px; border-radius:3px; }
  pre { background:var(--code-bg); border:1px solid var(--rule); border-left:3px solid var(--accent);
        padding:12px 14px; overflow-x:auto; border-radius:4px; margin:12px 0; }
  pre code { background:none; padding:0; font-size:12.5px; line-height:1.5; }

  table { border-collapse:collapse; width:100%; margin:14px 0; font-size:14.5px; }
  th,td { border:1px solid var(--rule); padding:7px 10px; text-align:left; vertical-align:top; }
  th { background:var(--accent-soft); font-weight:600; }
  td.num, th.num { text-align:right; font-family:var(--mono); white-space:nowrap; }
  tbody tr:nth-child(even) { background:#fafbfc; }

  .box { border-radius:5px; padding:13px 17px; margin:16px 0; border-left:4px solid; }
  .box p:first-child { margin-top:0; } .box p:last-child { margin-bottom:0; }
  .box .lbl { font-weight:700; text-transform:uppercase; font-size:11.5px;
              letter-spacing:.8px; display:block; margin-bottom:4px; }
  .warn{background:var(--warn-bg);border-color:var(--warn-br);}
  .bad {background:var(--bad-bg); border-color:var(--bad-br);}
  .good{background:var(--good-bg);border-color:var(--good-br);}
  .note{background:var(--accent-soft);border-color:var(--accent);}

  .stat-row { display:flex; flex-wrap:wrap; gap:12px; margin:18px 0; }
  .stat { flex:1 1 150px; border:1px solid var(--rule); border-radius:5px;
          padding:11px 13px; background:#fff; }
  .stat .v { font-size:25px; font-weight:700; font-family:var(--mono); letter-spacing:-1px; }
  .stat .k { font-size:11.5px; color:var(--muted); text-transform:uppercase;
             letter-spacing:.5px; margin-top:2px; }

  figure { margin:14px 0; }
  figure svg { display:block; width:100%; height:auto; }
  figcaption { font-size:12.5px; color:var(--muted); margin-top:7px; line-height:1.45; }
  figcaption b { color:var(--fg); }
  svg text { font-family:-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
  svg text.m { font-family:var(--mono); }

  .two { display:grid; grid-template-columns:1fr 1fr; gap:22px; }
  .kicker { font-size:19px; line-height:1.45; }
  .big { font-size:23px; line-height:1.4; font-weight:600; letter-spacing:-.3px; }
  .muted { color:var(--muted); }
  .rule { border:0; border-top:1px solid var(--rule); margin:16px 0; }
  .cite { font-size:12.5px; color:var(--muted); font-style:normal; }
  .cite a { color:var(--accent); text-decoration:none; }

  /* deck index */
  .index { width:1120px; margin:26px auto; background:#fff; border:1px solid #c9d2da;
           border-radius:6px; padding:30px 56px 36px; }
  .index ol { columns:2; column-gap:44px; font-size:14.5px; }
  .index a { color:var(--accent); text-decoration:none; }
  .index a:hover { text-decoration:underline; }

  @media print {
    html,body { background:#fff; }
    .index { display:none; }
    .slide { margin:0; border:0; border-radius:0; box-shadow:none;
             page-break-after:always; width:100%; min-height:0; height:100vh; }
    @page { size:1120px 630px; margin:0; }
  }
</style>
</head>
<body>
<svg width="0" height="0" style="position:absolute" aria-hidden="true" focusable="false"><defs>
  <marker id="ar"  markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
    <path d="M0,0 L7,3 L0,6 z" fill="#5a6472"/></marker>
  <marker id="ab"  markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
    <path d="M0,0 L7,3 L0,6 z" fill="#5a6472"/></marker>
  <marker id="arR" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
    <path d="M0,0 L7,3 L0,6 z" fill="#c0392b"/></marker>
  <marker id="ag"  markerWidth="9" markerHeight="9" refX="8" refY="3.2" orient="auto">
    <path d="M0,0 L8,3.2 L0,6.4 z" fill="#0b5fa5"/></marker>
</defs></svg>
'''
