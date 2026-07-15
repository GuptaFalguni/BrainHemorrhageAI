"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Download, FileText, Loader2 } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import {
  buildFullReportHtml,
  downloadTextFile,
  fetchAsDataUrl,
  modeLabelFromResults,
} from "@/lib/report/build-full-report";
import { useWorkspaceStore } from "@/stores/workspace-store";

function safeBaseName(fileName: string | null) {
  const raw = (fileName ?? "scan").replace(/\.(nii\.gz|nii)$/i, "");
  return raw.replace(/[^a-zA-Z0-9._-]+/g, "_").slice(0, 64) || "scan";
}

export function DownloadsSection() {
  const phase = useWorkspaceStore((s) => s.phase);
  const results = useWorkspaceStore((s) => s.results);
  const fileName = useWorkspaceStore((s) => s.fileName);
  const originalPreviewUrl = useWorkspaceStore((s) => s.originalPreviewUrl);
  const [busy, setBusy] = useState(false);

  if (phase !== "complete" || results.length === 0) return null;

  const handleDownload = async () => {
    setBusy(true);
    try {
      const overlayDataUrls: Record<string, string> = {};
      await Promise.all(
        results.map(async (r) => {
          if (!r.overlaySrc) return;
          const data = await fetchAsDataUrl(r.overlaySrc);
          overlayDataUrls[r.modelId] = data;
          if (r.predictionId) overlayDataUrls[r.predictionId] = data;
        }),
      );

      const html = buildFullReportHtml({
        fileName,
        modeLabel: modeLabelFromResults(results),
        results,
        generatedAt: new Date(),
        originalDataUrl: originalPreviewUrl,
        overlayDataUrls,
      });

      const name = `BrainHemorrhageAI_report_${safeBaseName(fileName)}.html`;
      downloadTextFile(name, html, "text/html;charset=utf-8");
      toast.success("Report downloaded", {
        description: "Includes original CT preview and live API overlays.",
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : "Download failed";
      toast.error(message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <section id="downloads" className="scroll-mt-20 px-4 py-10 sm:px-6">
      <div className="mx-auto max-w-3xl">
        <h2 className="mb-2 text-center text-2xl font-semibold tracking-tight">
          Download Report
        </h2>
        <p className="mb-8 text-center text-sm text-muted-foreground">
          One HTML file with original CT, prediction figures, and metrics.
        </p>

        <motion.div
          initial={{ opacity: 0, y: 10 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="glass-strong rounded-3xl p-6 sm:p-8"
        >
          <div className="flex flex-col gap-6 sm:flex-row sm:items-start">
            <div className="flex size-14 shrink-0 items-center justify-center rounded-2xl bg-primary/15 text-primary">
              <FileText className="size-6" />
            </div>
            <div className="min-w-0 flex-1">
              <h3 className="text-lg font-semibold">Full analysis report</h3>
              <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                Embeds the original CT preview and live prediction overlays from
                this run, plus volumes, confidence, timings, and the research
                disclaimer.
              </p>
              <Button
                type="button"
                size="lg"
                className="mt-6 h-12 w-full rounded-2xl sm:w-auto sm:min-w-[240px]"
                disabled={busy}
                onClick={handleDownload}
              >
                {busy ? (
                  <Loader2 className="size-4 animate-spin" />
                ) : (
                  <Download className="size-4" />
                )}
                {busy ? "Preparing…" : "Download full report"}
              </Button>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
