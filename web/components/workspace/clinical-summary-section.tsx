"use client";

import { motion } from "framer-motion";

import { COLOR_HINT } from "@/features/workspace/map-predict-result";
import { useWorkspaceStore, type ModelResult } from "@/stores/workspace-store";

function subtypeBlurb(result: ModelResult) {
  if (result.hemorrhageDetected === false || result.subtype === "None") {
    return "The model did not detect a hemorrhage pattern on this scan (research estimate only).";
  }
  const primary = result.subtypeVolumes.find((s) => s.isPrimary) ?? result.subtypeVolumes[0];
  if (!primary) {
    return "The model reported a hemorrhage finding. Review the overlay figure for details.";
  }
  const others = result.subtypeVolumes.filter((s) => !s.isPrimary);
  if (others.length === 0) {
    return `Primary prediction is ${primary.fullName} (${primary.code}) at ${primary.volumeMl.toFixed(2)} mL. Overlay color for ${primary.code} is ${COLOR_HINT[primary.code] ?? "shown in the legend"}.`;
  }
  const list = others
    .map((s) => `${s.code} ${s.volumeMl.toFixed(2)} mL`)
    .join(", ");
  return `Primary prediction (largest volume) is ${primary.fullName} (${primary.code}) at ${primary.volumeMl.toFixed(2)} mL. The model also predicted: ${list}. Match colors in the overlay legend — Red EDH, Yellow SDH, Cyan SAH, Green IPH, Magenta IVH.`;
}

function volumeBlurb(result: ModelResult) {
  return `Total predicted hemorrhage volume is about ${result.volumeMl.toFixed(1)} mL across all subtypes. Milliliters come from counting predicted voxels and converting with the scan’s spacing.`;
}

function confidenceBlurb(confidence: number) {
  const pct = Math.round(confidence * 100);
  if (pct >= 90) {
    return `The model assigned a high self-confidence of ${pct}%. That reflects how consistent its internal scores look — it does not mean an expert would agree ${pct}% of the time.`;
  }
  if (pct >= 70) {
    return `The model assigned a moderate self-confidence of ${pct}%. Review the overlay closely before drawing conclusions from the numbers alone.`;
  }
  return `The model assigned a lower self-confidence of ${pct}%. Prefer cautious interpretation and check the overlay carefully.`;
}

function speedBlurb(result: ModelResult) {
  return `${result.displayName} spent ${result.inferenceSeconds.toFixed(1)} s on network inference and ${result.processingSeconds.toFixed(1)} s end-to-end in this demo run.`;
}

function compareBlurb(results: ModelResult[]) {
  if (results.length < 2) return null;
  const names = results.map((r) => r.displayName);
  const span = (
    Math.max(...results.map((r) => r.volumeMl)) -
    Math.min(...results.map((r) => r.volumeMl))
  ).toFixed(1);
  const volumes = results
    .map((r) => `${r.displayName} ${r.volumeMl.toFixed(1)} mL`)
    .join("; ");
  const sameSubtype = results.every((r) => r.subtype === results[0].subtype);
  if (sameSubtype) {
    return `${names.join(" + ")} pointed to the same primary subtype (${results[0].subtype}). Total volume estimates span about ${span} mL (${volumes}).`;
  }
  const subtypes = results
    .map((r) => `${r.displayName}: ${r.subtype}`)
    .join("; ");
  return `The selected models disagreed on primary subtype (${subtypes}). Compare per-type volumes and overlays — disagreement is for review, not a clinical decision.`;
}

export function ClinicalSummarySection() {
  const phase = useWorkspaceStore((s) => s.phase);
  const results = useWorkspaceStore((s) => s.results);
  const fileName = useWorkspaceStore((s) => s.fileName);

  if (phase !== "complete" || results.length === 0) return null;

  const compare = compareBlurb(results);

  return (
    <section id="clinical" className="scroll-mt-20 px-4 py-10 sm:px-6">
      <div className="mx-auto max-w-3xl">
        <h2 className="mb-2 text-center text-2xl font-semibold tracking-tight">
          Clinical Summary
        </h2>
        <p className="mb-8 text-center text-sm text-muted-foreground">
          All predicted subtypes with volumes — primary is the largest.
        </p>

        <motion.article
          initial={{ opacity: 0, y: 10 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="glass-strong space-y-8 rounded-3xl p-6 sm:p-8"
        >
          <div>
            <p className="text-xs font-medium uppercase tracking-wider text-primary">
              Overview
            </p>
            <p className="mt-2 text-base leading-relaxed text-foreground/95 sm:text-lg">
              {fileName
                ? `For “${fileName}”, the AI listed every predicted hemorrhage subtype with volume.`
                : "The AI listed every predicted hemorrhage subtype with volume."}{" "}
              Research outputs only.
            </p>
          </div>

          {results.map((r) => (
            <div
              key={r.modelId}
              className="space-y-4 border-t border-border/50 pt-6 first:border-t-0 first:pt-0"
            >
              {results.length > 1 && (
                <p className="text-xs font-medium uppercase tracking-wider text-primary">
                  {r.displayName}
                </p>
              )}
              <p className="text-base leading-relaxed">{subtypeBlurb(r)}</p>

              {r.subtypeVolumes.length > 0 ? (
                <div className="overflow-hidden rounded-2xl border border-border/50">
                  <table className="w-full text-left text-sm">
                    <thead className="bg-surface-muted/50 text-xs uppercase tracking-wider text-muted-foreground">
                      <tr>
                        <th className="px-3 py-2 font-medium">Subtype</th>
                        <th className="px-3 py-2 font-medium">Volume</th>
                        <th className="px-3 py-2 font-medium">Share</th>
                        <th className="px-3 py-2 font-medium">Overlay</th>
                      </tr>
                    </thead>
                    <tbody>
                      {r.subtypeVolumes.map((row) => (
                        <tr
                          key={row.code}
                          className={
                            row.isPrimary
                              ? "bg-primary/10 font-medium"
                              : "border-t border-border/40"
                          }
                        >
                          <td className="px-3 py-2">
                            {row.code}
                            {row.isPrimary ? " · primary" : ""}
                            <div className="text-xs font-normal text-muted-foreground">
                              {row.fullName}
                            </div>
                          </td>
                          <td className="px-3 py-2 font-mono">
                            {row.volumeMl.toFixed(2)} mL
                          </td>
                          <td className="px-3 py-2 font-mono">
                            {row.percent.toFixed(1)}%
                          </td>
                          <td className="px-3 py-2 text-muted-foreground">
                            {COLOR_HINT[row.code] ?? "—"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : null}

              <p className="text-base leading-relaxed">{volumeBlurb(r)}</p>
              <p className="text-base leading-relaxed">{confidenceBlurb(r.confidence)}</p>
              <p className="text-sm text-muted-foreground">{speedBlurb(r)}</p>
            </div>
          ))}

          {compare && (
            <div className="border-t border-border/50 pt-6">
              <p className="text-xs font-medium uppercase tracking-wider text-primary">
                Putting the selected models together
              </p>
              <p className="mt-2 text-base leading-relaxed">{compare}</p>
            </div>
          )}

          <p className="rounded-2xl border border-disclaimer-border/40 bg-disclaimer/60 px-4 py-3 text-sm text-disclaimer-foreground">
            Research software only. Not for diagnosis, triage, or treatment
            decisions.
          </p>
        </motion.article>
      </div>
    </section>
  );
}
