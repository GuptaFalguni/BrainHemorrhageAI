"use client";

import { useCallback, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { Sparkles } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { ApiError } from "@/lib/api";
import { postPredict, fetchOverlayBlobUrl } from "@/lib/api/predict";
import { renderAxialPreview } from "@/lib/nifti/render-axial-slice";
import {
  mapPredictToResult,
  modelIdForMode,
} from "@/features/workspace/map-predict-result";
import {
  TIMELINE_STEPS,
  useWorkspaceStore,
  type TimelineStepId,
} from "@/stores/workspace-store";

export function AnalyzeSection() {
  const file = useWorkspaceStore((s) => s.file);
  const mode = useWorkspaceStore((s) => s.mode);
  const comparePreset = useWorkspaceStore((s) => s.comparePreset);
  const phase = useWorkspaceStore((s) => s.phase);
  const setPhase = useWorkspaceStore((s) => s.setPhase);
  const setTimelineStep = useWorkspaceStore((s) => s.setTimelineStep);
  const setElapsedMs = useWorkspaceStore((s) => s.setElapsedMs);
  const setResults = useWorkspaceStore((s) => s.setResults);
  const clearResults = useWorkspaceStore((s) => s.clearResults);
  const setError = useWorkspaceStore((s) => s.setError);
  const setOriginalPreviewUrl = useWorkspaceStore((s) => s.setOriginalPreviewUrl);
  const setProgressLabel = useWorkspaceStore((s) => s.setProgressLabel);
  const tick = useRef<ReturnType<typeof setInterval> | null>(null);
  const abortRef = useRef(false);

  const clearTick = useCallback(() => {
    if (tick.current) clearInterval(tick.current);
    tick.current = null;
  }, []);

  useEffect(() => () => {
    abortRef.current = true;
    clearTick();
  }, [clearTick]);

  const advanceTimeline = (step: TimelineStepId) => {
    setTimelineStep(step);
  };

  const runLive = async () => {
    if (!file) return;
    abortRef.current = false;
    clearTick();
    setError(null);
    setPhase("running");
    setElapsedMs(0);
    clearResults();
    setProgressLabel(null);
    advanceTimeline("upload");

    const start = Date.now();
    tick.current = setInterval(() => setElapsedMs(Date.now() - start), 250);

    requestAnimationFrame(() => {
      document.getElementById("progress")?.scrollIntoView({
        behavior: "smooth",
        block: "center",
      });
    });

    try {
      advanceTimeline("preprocessing");
      setProgressLabel("Reading CT and preparing original slice preview…");
      let previewUrl: string | null = null;
      try {
        const preview = await renderAxialPreview(file, 0.55);
        if (abortRef.current) return;
        previewUrl = preview.dataUrl;
        setOriginalPreviewUrl(preview.dataUrl);
      } catch {
        setOriginalPreviewUrl(null);
        toast.message("Original CT preview unavailable", {
          description: "Prediction will still run; overlay comes from the API.",
        });
      }

      const modelIds = modelIdForMode(mode, comparePreset);
      const results = [];

      for (let i = 0; i < modelIds.length; i++) {
        const modelId = modelIds[i];
        const label =
          modelIds.length > 1
            ? `Running model ${i + 1} of ${modelIds.length} (${modelId})…`
            : `Running ${modelId}…`;
        setProgressLabel(
          modelId.includes("nnunet")
            ? `${label} nnU-Net may take several minutes on CPU.`
            : label,
        );
        advanceTimeline("segmentation");

        const response = await postPredict(file, modelId);
        if (abortRef.current) return;

        advanceTimeline("volume");
        advanceTimeline("clinical");

        const mapped = mapPredictToResult(response);
        mapped.originalSrc = previewUrl;
        try {
          const { overlayUrl, overlay3dUrl } = await fetchOverlayBlobUrl(
            response.download_urls,
          );
          if (overlayUrl) mapped.overlaySrc = overlayUrl;
          if (overlay3dUrl) mapped.overlay3dSrc = overlay3dUrl;
        } catch (overlayErr) {
          const msg =
            overlayErr instanceof Error
              ? overlayErr.message
              : "Could not load overlay image";
          toast.message("Overlay image issue", { description: msg });
        }
        results.push(mapped);
      }

      advanceTimeline("completed");
      clearTick();
      setElapsedMs(Date.now() - start);
      setProgressLabel(null);
      setResults(results);
      toast.success("Analysis complete", {
        description: "Real CT overlays loaded from the API.",
      });
      document.getElementById("results")?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    } catch (err) {
      clearTick();
      setElapsedMs(Date.now() - start);
      setTimelineStep(TIMELINE_STEPS[0].id);
      const message =
        err instanceof ApiError
          ? err.message
          : err instanceof Error
            ? err.message
            : "Analysis failed";
      setError(message);
      setProgressLabel(null);
      toast.error("Analysis failed", { description: message });
    }
  };

  const disabled = !file || phase === "running";

  return (
    <section id="analyze" className="scroll-mt-20 px-4 py-6 sm:px-6">
      <div className="mx-auto flex max-w-3xl flex-col items-center justify-center gap-3">
        <motion.div
          whileHover={disabled ? undefined : { scale: 1.02 }}
          whileTap={disabled ? undefined : { scale: 0.98 }}
        >
          <Button
            type="button"
            size="lg"
            disabled={disabled}
            onClick={runLive}
            className="h-14 min-w-[220px] rounded-2xl px-10 text-base font-semibold shadow-[0_0_40px_-10px_var(--primary)]"
            aria-busy={phase === "running"}
          >
            <Sparkles className="size-5" />
            {phase === "running" ? "Analyzing…" : "Analyze scan"}
          </Button>
        </motion.div>
        {!file && (
          <p className="text-center text-xs text-muted-foreground">
            Select a CT file to enable Analyze.
          </p>
        )}
        {mode === "nnunet" || mode === "ssl" || mode === "compare" ? (
          <p className="max-w-md text-center text-xs text-muted-foreground">
            {mode === "compare"
              ? "Compare runs your selected models one after another. nnU-Net and SSL can take several minutes each on CPU."
              : "nnU-Net and SSL nnU-Net run on the real API and can take several minutes on CPU."}{" "}
            Keep this tab open.
          </p>
        ) : null}
      </div>
    </section>
  );
}
