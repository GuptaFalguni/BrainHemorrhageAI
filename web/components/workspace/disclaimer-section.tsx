"use client";

import { ShieldAlert } from "lucide-react";

export function DisclaimerSection() {
  return (
    <section id="disclaimer" className="scroll-mt-20 px-4 py-10 pb-20 sm:px-6">
      <div className="mx-auto max-w-3xl">
        <div className="glass-strong rounded-3xl border border-disclaimer-border/50 p-8 sm:p-10">
          <div className="mb-5 flex size-12 items-center justify-center rounded-2xl bg-disclaimer text-disclaimer-foreground">
            <ShieldAlert className="size-6" />
          </div>
          <h2 className="text-2xl font-semibold tracking-tight">
            Research Disclaimer
          </h2>
          <ul className="mt-6 space-y-3 text-base leading-relaxed text-muted-foreground">
            <li>
              <strong className="text-foreground">Research Software</strong> —
              this application is an investigational demonstration platform.
            </li>
            <li>
              <strong className="text-foreground">Not for Clinical Use</strong> —
              do not use outputs in patient care workflows.
            </li>
            <li>
              <strong className="text-foreground">No diagnosis</strong> — the AI
              does not replace imaging interpretation by a qualified clinician.
            </li>
            <li>
              <strong className="text-foreground">No treatment advice</strong> —
              never adjust therapy based on these results.
            </li>
          </ul>
        </div>
      </div>
    </section>
  );
}
