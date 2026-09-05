"use client";

import { motion } from "framer-motion";

import { MetricInfo } from "@/components/ui/metric-info";
import { METRIC_TOOLTIPS } from "@/content/explainers";
import { type ModelResult, useWorkspaceStore } from "@/stores/workspace-store";
import { cn } from "@/lib/utils";

function ResultCard({
  result,
  originalSrc,
}: {
  result: ModelResult;
  originalSrc: string | null;
}) {
  return (
    <motion.article
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-strong flex flex-col overflow-hidden rounded-3xl p-5 sm:p-6"
    >
      <div className="mb-4">
        <p className="text-xs font-medium uppercase tracking-wider text-primary">
          {result.displayName}
        </p>
        <h3 className="mt-1 text-xl font-semibold">{result.subtypeFull}</h3>
        <p className="text-sm text-muted-foreground">
          Primary finding (largest volume): {result.subtype}
        </p>
      </div>

      <div className="grid gap-3 lg:grid-cols-2">
        <div className="overflow-hidden rounded-2xl border border-border/50 bg-[#020617]">
          <p className="border-b border-border/40 px-3 py-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
            Original CT (your upload)
          </p>
          <div className="flex min-h-[240px] items-center justify-center p-3">
            {originalSrc || result.originalSrc ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={originalSrc || result.originalSrc || ""}
                alt="Original CT axial slice"
                className="max-h-[400px] w-full object-contain"
              />
            ) : (
              <p className="text-sm text-muted-foreground">CT preview unavailable</p>
            )}
          </div>
        </div>

        <div className="overflow-hidden rounded-2xl border border-border/50 bg-[#020617]">
          <p className="border-b border-border/40 px-3 py-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
            Predicted segmentation (from API)
          </p>
          <div className="flex min-h-[240px] items-center justify-center overflow-auto bg-[#0b1220] p-3">
            {result.overlaySrc ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={result.overlaySrc}
                alt={`${result.displayName} prediction overlay`}
                className="h-auto w-full max-h-[480px] object-contain"
                style={{ background: "#fff" }}
              />
            ) : (
              <p className="px-4 text-center text-sm text-muted-foreground">
                No overlay returned. Confirm the API finished this run.
              </p>
            )}
          </div>
          <p className="border-t border-border/40 px-3 py-2 text-[11px] leading-relaxed text-muted-foreground">
            This figure is produced by the model pipeline: Original CT · Prediction
            overlay · Class map on the hemorrhage slice.
          </p>
        </div>
      </div>

      {result.overlay3dSrc ? (
        <div className="mt-3 overflow-hidden rounded-2xl border border-border/50 bg-[#020617]">
          <p className="border-b border-border/40 px-3 py-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
            Orthogonal views + MIP
          </p>
          <div className="p-3">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={result.overlay3dSrc}
              alt={`${result.displayName} orthogonal overlays`}
              className="max-h-[420px] w-full object-contain"
            />
          </div>
        </div>
      ) : null}

      <dl className="mt-5 grid grid-cols-2 gap-3 text-sm">
        <div className="rounded-2xl bg-surface-muted/60 p-3">
          <dt className="flex items-center gap-1 text-xs text-muted-foreground">
            Total volume
            <MetricInfo label="Volume" explanation={METRIC_TOOLTIPS.volume} />
          </dt>
          <dd className="mt-1 font-mono text-lg font-medium">
            {result.volumeMl.toFixed(1)} mL
          </dd>
        </div>
        <div className="rounded-2xl bg-surface-muted/60 p-3">
          <dt className="flex items-center gap-1 text-xs text-muted-foreground">
            Confidence
            <MetricInfo
              label="Confidence"
              explanation={METRIC_TOOLTIPS.confidence}
            />
          </dt>
          <dd className="mt-1 font-mono text-lg font-medium">
            {(result.confidence * 100).toFixed(0)}%
          </dd>
        </div>
        <div className="rounded-2xl bg-surface-muted/60 p-3">
          <dt className="flex items-center gap-1 text-xs text-muted-foreground">
            Inference time
            <MetricInfo
              label="Inference time"
              explanation={METRIC_TOOLTIPS.inference_time}
            />
          </dt>
          <dd className="mt-1 font-mono text-base">
            {result.inferenceSeconds.toFixed(1)} s
          </dd>
        </div>
        <div className="rounded-2xl bg-surface-muted/60 p-3">
          <dt className="flex items-center gap-1 text-xs text-muted-foreground">
            Processing time
            <MetricInfo
              label="Processing time"
              explanation={METRIC_TOOLTIPS.processing_time}
            />
          </dt>
          <dd className="mt-1 font-mono text-base">
            {result.processingSeconds.toFixed(1)} s
          </dd>
        </div>
      </dl>

      {result.subtypeVolumes.length > 0 ? (
        <div className="mt-4 overflow-hidden rounded-2xl border border-border/50">
          <div className="border-b border-border/40 bg-surface-muted/40 px-4 py-2.5">
            <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
              All predicted subtypes (by volume)
            </p>
            <p className="mt-0.5 text-[11px] text-muted-foreground">
              Primary = largest volume. Overlay colors: Red EDH · Yellow SDH · Cyan
              SAH · Green IPH · Magenta IVH
            </p>
          </div>
          <ul className="divide-y divide-border/40">
            {result.subtypeVolumes.map((row) => (
              <li
                key={row.code}
                className={cn(
                  "flex items-center justify-between gap-3 px-4 py-3 text-sm",
                  row.isPrimary && "bg-primary/10",
                )}
              >
                <div className="min-w-0">
                  <p className="font-medium">
                    {row.code}
                    {row.isPrimary ? (
                      <span className="ml-2 rounded-full bg-primary/20 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-primary">
                        Primary
                      </span>
                    ) : null}
                  </p>
                  <p className="truncate text-xs text-muted-foreground">
                    {row.fullName}
                  </p>
                </div>
                <div className="shrink-0 text-right font-mono text-sm">
                  <p>{row.volumeMl.toFixed(2)} mL</p>
                  <p className="text-xs text-muted-foreground">
                    {row.percent.toFixed(1)}%
                  </p>
                </div>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </motion.article>
  );
}

export function ResultsSection() {
  const phase = useWorkspaceStore((s) => s.phase);
  const results = useWorkspaceStore((s) => s.results);
  const originalPreviewUrl = useWorkspaceStore((s) => s.originalPreviewUrl);

  if (phase !== "complete" || results.length === 0) return null;

  return (
    <section id="results" className="scroll-mt-20 px-4 py-10 sm:px-6">
      <div className="mx-auto max-w-5xl">
        <h2 className="mb-2 text-center text-2xl font-semibold tracking-tight">
          Prediction Results
        </h2>
        <p className="mb-8 text-center text-sm text-muted-foreground">
          Live Original CT vs model prediction figures from the inference API.
        </p>
        <div
          className={cn(
            "grid gap-6",
            results.length > 1 ? "grid-cols-1" : "mx-auto max-w-4xl",
          )}
        >
          {results.map((r) => (
            <ResultCard
              key={r.predictionId ?? r.modelId}
              result={r}
              originalSrc={originalPreviewUrl}
            />
          ))}
        </div>
      </div>
    </section>
  );
}
