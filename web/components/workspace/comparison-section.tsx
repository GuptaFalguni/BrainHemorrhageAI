"use client";

import { motion } from "framer-motion";

import { MetricInfo } from "@/components/ui/metric-info";
import { COMPARISON_ROWS, type ComparisonWinner } from "@/content/explainers";
import { cn } from "@/lib/utils";

const WINNER_LABEL: Record<ComparisonWinner, string> = {
  monai: "MONAI",
  nnunet: "nnU-Net",
  ssl: "nnU-Net Ensemble",
  semi: "SSL",
  tie: "—",
};

export function ComparisonSection() {
  return (
    <section id="comparison" className="scroll-mt-20 px-4 py-10 sm:px-6">
      <div className="mx-auto max-w-6xl">
        <h2 className="mb-2 text-center text-2xl font-semibold tracking-tight">
          Model Comparison
        </h2>
        <p className="mb-8 text-center text-sm text-muted-foreground">
          How the four models differ — explained in plain language.
        </p>

        <motion.div
          initial={{ opacity: 0, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="glass-strong overflow-hidden rounded-3xl"
        >
          <div className="overflow-x-auto">
            <table className="w-full min-w-[960px] text-left text-sm">
              <thead>
                <tr className="border-b border-border/60 bg-surface-muted/40">
                  <th className="px-4 py-4 font-medium">Metric</th>
                  <th className="px-4 py-4 font-medium">MONAI</th>
                  <th className="px-4 py-4 font-medium">nnU-Net</th>
                  <th className="px-4 py-4 font-medium">nnU-Net Ensemble</th>
                  <th className="px-4 py-4 font-medium">SSL</th>
                  <th className="px-4 py-4 font-medium">Meaning</th>
                  <th className="px-4 py-4 font-medium">Winner</th>
                </tr>
              </thead>
              <tbody>
                {COMPARISON_ROWS.map((row) => (
                  <tr
                    key={row.id}
                    className="border-b border-border/40 last:border-0"
                  >
                    <td className="px-4 py-4">
                      <span className="inline-flex items-center gap-1.5 font-medium">
                        {row.metric}
                        <MetricInfo label={row.metric} explanation={row.tooltip} />
                      </span>
                    </td>
                    <td
                      className={cn(
                        "px-4 py-4 font-mono text-xs sm:text-sm",
                        row.winner === "monai" && "text-primary",
                      )}
                    >
                      {row.monai}
                    </td>
                    <td
                      className={cn(
                        "px-4 py-4 font-mono text-xs sm:text-sm",
                        row.winner === "nnunet" && "text-primary",
                      )}
                    >
                      {row.nnunet}
                    </td>
                    <td
                      className={cn(
                        "px-4 py-4 font-mono text-xs sm:text-sm",
                        row.winner === "ssl" && "text-primary",
                      )}
                    >
                      {row.ssl}
                    </td>
                    <td
                      className={cn(
                        "px-4 py-4 font-mono text-xs sm:text-sm",
                        row.winner === "semi" && "text-primary",
                      )}
                    >
                      {row.semi}
                    </td>
                    <td className="max-w-[220px] px-4 py-4 text-muted-foreground">
                      {row.meaning}
                    </td>
                    <td className="px-4 py-4">
                      <span
                        className={cn(
                          "rounded-full px-2.5 py-1 text-xs font-medium",
                          row.winner === "tie"
                            ? "bg-muted text-muted-foreground"
                            : "bg-primary/15 text-primary",
                        )}
                      >
                        {WINNER_LABEL[row.winner]}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="border-t border-border/40 px-4 py-3 text-xs leading-relaxed text-muted-foreground">
            nnU-Net Ensemble averages two folds on the locked 29-case test (Dice
            0.489 · IoU 0.343 · volume error 18.5 mL mean). SSL means
            semi-supervised learning: one nnU-Net trained on labeled CTs plus
            teacher masks (Dice 0.471). IoU and volume error were not scored for
            SSL, so those cells are blank. Lower volume error belongs to V1
            nnU-Net. Live Compare runs only the models you pick.
          </p>
        </motion.div>
      </div>
    </section>
  );
}
