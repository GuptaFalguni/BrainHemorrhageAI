"use client";

import { motion } from "framer-motion";
import { Check, FlaskConical, Gauge, Layers2 } from "lucide-react";

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
    points: ["Highest accuracy", "Research quality", "Minutes on CPU"],
    icon: FlaskConical,
  },
  {
    id: "compare",
    title: "Compare Both",
    badge: "Side-by-side",
    points: ["Run both sequentially", "Longer wait", "Direct comparison"],
    icon: Layers2,
  },
];

export function ModelSelectSection() {
  const mode = useWorkspaceStore((s) => s.mode);
  const setMode = useWorkspaceStore((s) => s.setMode);
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
          className="grid gap-4 md:grid-cols-3"
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
      </div>
    </section>
  );
}
