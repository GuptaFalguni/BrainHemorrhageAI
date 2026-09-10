"use client";

import { AnimatePresence, motion } from "framer-motion";
import { Check } from "lucide-react";

import {
  TIMELINE_STEPS,
  useWorkspaceStore,
  type TimelineStepId,
} from "@/stores/workspace-store";
import { cn } from "@/lib/utils";

function stepIndex(id: TimelineStepId) {
  return TIMELINE_STEPS.findIndex((s) => s.id === id);
}

function formatElapsed(ms: number) {
  const s = Math.floor(ms / 1000);
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return `${String(m).padStart(2, "0")}:${String(rem).padStart(2, "0")}`;
}

export function ProgressSection() {
  const phase = useWorkspaceStore((s) => s.phase);
  const step = useWorkspaceStore((s) => s.timelineStep);
  const elapsedMs = useWorkspaceStore((s) => s.elapsedMs);
  const mode = useWorkspaceStore((s) => s.mode);
  const errorMessage = useWorkspaceStore((s) => s.errorMessage);
  const progressLabel = useWorkspaceStore((s) => s.progressLabel);

  const visible = phase === "running" || phase === "complete" || phase === "error";
  const current = stepIndex(step);

  if (!visible) return null;

  return (
    <section id="progress" className="scroll-mt-20 px-4 py-10 sm:px-6">
      <div className="mx-auto max-w-3xl">
        <h2 className="mb-2 text-center text-2xl font-semibold tracking-tight">
          Live Analysis Timeline
        </h2>
        <p className="mb-8 text-center text-sm text-muted-foreground">
          {mode === "compare"
            ? "Running your selected models one after another via the live API (nnU-Net models may take several minutes each)."
            : mode === "ssl" || mode === "nnunet" || mode === "semi"
              ? "Live progress while the API runs nnU-Net (may take several minutes on CPU)."
              : "Live progress while the API runs real inference on your CT."}
        </p>

        <div className="glass-strong rounded-3xl p-6 sm:p-8">
          <div className="mb-6 flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Elapsed</span>
            <span className="font-mono text-primary">{formatElapsed(elapsedMs)}</span>
          </div>
          {progressLabel ? (
            <p className="mb-6 text-sm text-primary" role="status">
              {progressLabel}
            </p>
          ) : null}

          <ol className="space-y-0">
            {TIMELINE_STEPS.map((item, index) => {
              const done = index < current || phase === "complete";
              const active = index === current && phase === "running";
              return (
                <li key={item.id} className="relative flex gap-4 pb-6 last:pb-0">
                  {index < TIMELINE_STEPS.length - 1 && (
                    <span
                      aria-hidden
                      className={cn(
                        "absolute left-[15px] top-8 h-[calc(100%-1.5rem)] w-px",
                        done || active ? "bg-primary/50" : "bg-border",
                      )}
                    />
                  )}
                  <span
                    className={cn(
                      "relative z-10 mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-full border text-xs font-medium",
                      done && "border-primary bg-primary text-primary-foreground",
                      active && "border-primary bg-primary/20 text-primary",
                      !done && !active && "border-border bg-surface-muted text-muted-foreground",
                    )}
                  >
                    {done && !active ? (
                      <Check className="size-4" />
                    ) : (
                      <span>{index + 1}</span>
                    )}
                    {active && (
                      <motion.span
                        className="absolute inset-0 rounded-full border border-primary"
                        animate={{ scale: [1, 1.35], opacity: [0.6, 0] }}
                        transition={{ repeat: Infinity, duration: 1.4 }}
                      />
                    )}
                  </span>
                  <div className="pt-1">
                    <p
                      className={cn(
                        "text-sm font-medium",
                        active || done ? "text-foreground" : "text-muted-foreground",
                      )}
                    >
                      {item.label}
                    </p>
                    {active && (
                      <motion.p
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="mt-0.5 text-xs text-primary"
                      >
                        In progress…
                      </motion.p>
                    )}
                  </div>
                </li>
              );
            })}
          </ol>

          <AnimatePresence>
            {errorMessage && (
              <motion.p
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="mt-6 rounded-2xl border border-danger/40 bg-danger/10 px-4 py-3 text-sm text-danger"
                role="alert"
              >
                {errorMessage}
              </motion.p>
            )}
          </AnimatePresence>
        </div>
      </div>
    </section>
  );
}
