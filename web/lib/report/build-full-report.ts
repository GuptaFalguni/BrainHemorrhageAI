import type { ModelResult } from "@/stores/workspace-store";

export type FullReportInput = {
  fileName: string | null;
  modeLabel: string;
  results: ModelResult[];
  generatedAt: Date;
  originalDataUrl?: string | null;
  overlayDataUrls?: Record<string, string>;
};

function escapeHtml(value: string) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function explainSubtype(full: string, code: string) {
  if (code === "SDH") {
    return `${full} (${code}) means blood under the dura on the outer brain surface. The prediction figure shows where the model highlighted this pattern.`;
  }
  if (code === "None" || !code) {
    return "The model did not report a hemorrhage subtype for this run.";
  }
  return `${full} (${code}) is the hemorrhage pattern the model most strongly highlighted on this scan.`;
}

function confidenceBand(confidence: number) {
  const pct = Math.round(confidence * 100);
  if (pct >= 90) {
    return `The model scored its own certainty at ${pct}%. That is a research self-score — not a guarantee that an expert would agree.`;
  }
  if (pct >= 70) {
    return `The model scored its own certainty at ${pct}%. Treat moderate scores as a signal to review the overlay carefully.`;
  }
  return `The model scored its own certainty at ${pct}%. Lower scores mean the result should be interpreted with extra caution.`;
}

function buildModelBlock(
  result: ModelResult,
  originalDataUrl: string | null | undefined,
  overlayDataUrl: string | undefined,
) {
  const vol = result.volumeMl.toFixed(1);
  const conf = Math.round(result.confidence * 100);
  const originalBlock = originalDataUrl
    ? `<img src="${originalDataUrl}" alt="Original CT" style="max-width:100%;border-radius:12px;background:#020617" />`
    : `<p class="muted">Original CT preview not available.</p>`;
  const overlayBlock = overlayDataUrl
    ? `<img src="${overlayDataUrl}" alt="${escapeHtml(result.displayName)} prediction" style="max-width:100%;border-radius:12px;background:#020617" />`
    : `<p class="muted">Prediction overlay not available.</p>`;

  return `
  <section class="card">
    <h2>${escapeHtml(result.displayName)}</h2>
    <p class="muted">Model ID: ${escapeHtml(result.modelId)}${
      result.predictionId ? ` · Prediction: ${escapeHtml(result.predictionId)}` : ""
    }</p>
    <div style="display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));margin:16px 0">
      <figure>
        ${originalBlock}
        <figcaption>Original CT (upload preview)</figcaption>
      </figure>
      <figure>
        ${overlayBlock}
        <figcaption>Predicted segmentation figure from API</figcaption>
      </figure>
    </div>
    <h3>What the model found</h3>
    <p>${escapeHtml(explainSubtype(result.subtypeFull, result.subtype))}</p>
    <p>Estimated total hemorrhage volume: <strong>${vol} mL</strong>.</p>
    <p>${escapeHtml(confidenceBand(result.confidence))}</p>
    ${
      result.subtypeVolumes?.length
        ? `<h3>All predicted subtypes</h3>
    <table>
      <thead><tr><th>Subtype</th><th>Volume</th><th>Share</th><th>Role</th></tr></thead>
      <tbody>
        ${result.subtypeVolumes
          .map(
            (row) =>
              `<tr><td>${escapeHtml(row.code)} — ${escapeHtml(row.fullName)}</td><td>${row.volumeMl.toFixed(2)} mL</td><td>${row.percent.toFixed(1)}%</td><td>${row.isPrimary ? "Primary (max volume)" : ""}</td></tr>`,
          )
          .join("")}
      </tbody>
    </table>`
        : ""
    }
    <table>
      <tbody>
        <tr><th>Primary finding</th><td>${escapeHtml(result.subtypeFull)} (${escapeHtml(result.subtype)})</td></tr>
        <tr><th>Total volume</th><td>${vol} mL</td></tr>
        <tr><th>Confidence</th><td>${conf}%</td></tr>
        <tr><th>Inference time</th><td>${result.inferenceSeconds.toFixed(1)} s</td></tr>
        <tr><th>Total processing</th><td>${result.processingSeconds.toFixed(1)} s</td></tr>
      </tbody>
    </table>
  </section>`;
}

function buildCompareNote(results: ModelResult[]) {
  if (results.length < 2) return "";
  const [a, b] = results;
  const delta = Math.abs(a.volumeMl - b.volumeMl).toFixed(1);
  const faster =
    a.inferenceSeconds <= b.inferenceSeconds ? a.displayName : b.displayName;
  const same = a.subtype === b.subtype;
  return `
  <section class="card">
    <h2>Side-by-side note</h2>
    <p>${
      same
        ? `Both models predicted the same subtype (${escapeHtml(a.subtype)}).`
        : `Subtype differed (${escapeHtml(a.subtype)} vs ${escapeHtml(b.subtype)}).`
    }
    Estimated volumes differ by about <strong>${delta} mL</strong>
    (${a.volumeMl.toFixed(1)} vs ${b.volumeMl.toFixed(1)}).
    ${escapeHtml(faster)} finished faster in this run.</p>
  </section>`;
}

export function buildFullReportHtml(input: FullReportInput): string {
  const when = input.generatedAt.toLocaleString();
  const file = input.fileName ?? "unnamed-scan";
  const blocks = input.results
    .map((r) =>
      buildModelBlock(
        r,
        input.originalDataUrl,
        input.overlayDataUrls?.[r.modelId] ??
          input.overlayDataUrls?.[r.predictionId ?? ""],
      ),
    )
    .join("\n");

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>BrainHemorrhageAI Report — ${escapeHtml(file)}</title>
  <style>
    :root { color-scheme: light dark; --fg: #0f1720; --muted: #5b6b7c; --border: #d5dee6; --card: #ffffff; --bg: #f4f6f8; }
    @media (prefers-color-scheme: dark) {
      :root { --fg: #eef3f7; --muted: #94a3b4; --border: #24303a; --card: #0e1318; --bg: #050708; }
    }
    body { font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 0; background: var(--bg); color: var(--fg); line-height: 1.55; }
    main { max-width: 900px; margin: 0 auto; padding: 32px 20px 64px; }
    h1 { font-size: 1.75rem; margin: 0 0 8px; }
    h2 { font-size: 1.25rem; margin: 0 0 8px; }
    h3 { font-size: 1.05rem; margin: 20px 0 8px; }
    .muted { color: var(--muted); font-size: 0.95rem; }
    .banner { border: 1px solid #b45309; background: #fff7ed; color: #78350f; padding: 14px 16px; border-radius: 12px; margin: 20px 0 28px; }
    @media (prefers-color-scheme: dark) {
      .banner { background: #292524; color: #fde68a; border-color: #a16207; }
    }
    .card { background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 20px; margin-bottom: 20px; }
    figcaption { color: var(--muted); font-size: 0.85rem; margin-top: 8px; }
    table { width: 100%; border-collapse: collapse; margin-top: 12px; }
    th, td { text-align: left; padding: 8px 10px; border-top: 1px solid var(--border); vertical-align: top; }
    th { width: 36%; color: var(--muted); font-weight: 600; }
    footer { margin-top: 28px; font-size: 0.85rem; color: var(--muted); }
  </style>
</head>
<body>
  <main>
    <h1>BrainHemorrhageAI — Analysis Report</h1>
    <p class="muted">Research segmentation summary with live CT figures.</p>
    <div class="banner">
      <strong>Research Software — Not for Clinical Use.</strong>
      This report is not a diagnosis and must not guide treatment.
    </div>
    <section class="card">
      <div><strong>Scan file:</strong> ${escapeHtml(file)}</div>
      <div><strong>Analysis mode:</strong> ${escapeHtml(input.modeLabel)}</div>
      <div><strong>Generated:</strong> ${escapeHtml(when)}</div>
    </section>
    ${blocks}
    ${buildCompareNote(input.results)}
    <section class="card">
      <h2>How to read these numbers</h2>
      <p><strong>Volume</strong> — estimated size of the highlighted hemorrhage region in milliliters.</p>
      <p><strong>Confidence</strong> — how certain the model appears about its own output. High confidence is not clinical proof.</p>
      <p><strong>Inference / processing time</strong> — how long this run took on the server.</p>
    </section>
    <footer>
      Generated by BrainHemorrhageAI. Open in a browser. Use Print → Save as PDF if needed.
    </footer>
  </main>
</body>
</html>`;
}

export function downloadTextFile(filename: string, content: string, mime: string) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export async function fetchAsDataUrl(src: string): Promise<string> {
  const response = await fetch(src);
  if (!response.ok) {
    throw new Error(`Failed to load image (${response.status})`);
  }
  const blob = await response.blob();
  return await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error("Failed to encode image"));
    reader.readAsDataURL(blob);
  });
}

export function modeLabelFromResults(results: ModelResult[]): string {
  if (results.length > 1) return "Compare Both (MONAI + nnU-Net)";
  return results[0]?.displayName ?? "Single model";
}
