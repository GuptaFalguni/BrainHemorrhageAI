"use client";

import { motion } from "framer-motion";
import { Check, FlaskConical, Gauge, Layers2, Sparkles } from "lucide-react";

import {
  COMPARE_PRESETS,
  type ComparePresetId,
} from "@/features/workspace/compare-presets";
import { type ModelMode, useWorkspaceStore } from "@/stores/workspace-store";
import { cn } from "@/lib/utils";

const CHOICES: {
  id: ModelMode;
  title: string;
  badge: string;
  points: string[];
  icon: typeof Gauge;
}[] = [
  {
    id: "monai",
    title: "MONAI",
    badge: "Interactive",
    points: ["Fast", "Interactive demos", "~20 s on CPU"],
    icon: Gauge,
  },
  {
    id: "nnunet",
    title: "nnU-Net",
    badge: "Research",
    points: ["V1 comparator", "Research quality", "Minutes on CPU"],
    icon: FlaskConical,
  },
  {
    id: "ssl",
    title: "SSL nnU-Net",
    badge: "Research",
    points: ["2-fold ensemble", "Best fold 0.4965", "Minutes on CPU"],
    icon: Sparkles,
  },
  {
    id: "compare",
    title: "Compare",
    badge: "Side-by-side",
    points: ["Pick 2 or all 3", "Run sequentially", "Shared report"],
    icon: Layers2,
  },
];

export function ModelSelectSection() {
  const mode = useWorkspaceStore((s) => s.mode);
  const setMode = useWorkspaceStore((s) => s.setMode);
  const comparePreset = useWorkspaceStore((s) => s.comparePreset);
  const setComparePreset = useWorkspaceStore((s) => s.setComparePreset);
  const phase = useWorkspaceStore((s) => s.phase);
  const locked = phase === "running";

  return (
    <section id="models" className="scroll-mt-20 px-4 py-10 sm:px-6">
      <div className="mx-auto max-w-5xl">
        <h2 className="mb-2 text-center text-2xl font-semibold tracking-tight">
          Model Selection
        </h2>
        <p className="mb-8 text-center text-sm text-muted-foreground">
          Choose one option. Only one selection at a time.
        </p>

        <div
          className="grid gap-4 md:grid-cols-2 xl:grid-cols-4"
          role="radiogroup"
          aria-label="Model selection"
        >
          {CHOICES.map((choice, i) => {
            const selected = mode === choice.id;
            const Icon = choice.icon;
            return (
              <motion.button
                key={choice.id}
                type="button"
                role="radio"
                aria-checked={selected}
                disabled={locked}
                initial={{ opacity: 0, y: 12 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.05 }}
                whileHover={locked ? undefined : { y: -2 }}
                onClick={() => setMode(choice.id)}
                className={cn(
                  "glass relative rounded-3xl p-6 text-left transition-shadow",
                  selected
                    ? "ring-2 ring-primary shadow-[0_0_48px_-16px_var(--primary)]"
                    : "hover:border-primary/40",
                  locked && "opacity-70",
                )}
              >
                {selected && (
                  <span className="absolute right-4 top-4 flex size-6 items-center justify-center rounded-full bg-primary text-primary-foreground">
                    <Check className="size-3.5" aria-hidden />
                  </span>
                )}
                <div className="mb-4 flex size-11 items-center justify-center rounded-2xl bg-primary/15 text-primary">
                  <Icon className="size-5" />
                </div>
                <p className="text-lg font-semibold">{choice.title}</p>
                <p className="mt-1 text-xs font-medium uppercase tracking-wider text-primary">
                  {choice.badge}
                </p>
                <ul className="mt-4 space-y-1.5 text-sm text-muted-foreground">
                  {choice.points.map((p) => (
                    <li key={p}>{p}</li>
                  ))}
                </ul>
            </motion.button>
          );
        })}
        </div>

        {mode === "compare" ? (
          <div className="mx-auto mt-8 max-w-xl">
            <label
              htmlFor="compare-preset"
              className="mb-2 block text-center text-sm font-medium"
            >
              Models to compare
            </label>
            <p className="mb-3 text-center text-xs text-muted-foreground">
              Choose all three, or any pair. Analyze and the report use this set.
            </p>
            <select
              id="compare-preset"
              value={comparePreset}
              disabled={locked}
              onChange={(e) =>
                setComparePreset(e.target.value as ComparePresetId)
              }
              className="h-12 w-full rounded-2xl border border-border bg-surface px-4 text-sm outline-none ring-offset-background focus-visible:ring-2 focus-visible:ring-primary disabled:opacity-70"
            >
              {COMPARE_PRESETS.map((preset) => (
                <option key={preset.id} value={preset.id}>
                  {preset.label}
                </option>
              ))}
            </select>
          </div>
        ) : null}
      </div>
    </section>
  );
}
