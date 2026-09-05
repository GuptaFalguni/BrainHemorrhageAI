"use client";

import { motion } from "framer-motion";

import { RESEARCH_DISCLAIMER } from "@/lib/constants";

export function HeroSection() {
  return (
    <section id="hero" className="relative overflow-hidden pt-16 pb-10 sm:pt-24 sm:pb-14">
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 -z-10"
        style={{
          background:
            "radial-gradient(ellipse 80% 50% at 50% -20%, color-mix(in oklab, var(--primary) 22%, transparent), transparent 70%)",
        }}
      />
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
        className="mx-auto max-w-3xl px-4 text-center sm:px-6"
      >
        <p className="mb-4 text-xs font-medium uppercase tracking-[0.2em] text-primary">
          Research software
        </p>
        <h1 className="font-display text-4xl font-semibold tracking-tight text-foreground sm:text-6xl">
          BrainHemorrhageAI
        </h1>
        <p className="mt-5 text-lg text-muted-foreground sm:text-xl">
          AI-powered Brain Hemorrhage Segmentation
          <br className="hidden sm:block" /> and Volume Analysis
        </p>
        <p className="mx-auto mt-6 max-w-2xl text-sm leading-relaxed text-muted-foreground/90 sm:text-base">
          Upload a CT scan, choose a model, and explore segmentation overlays,
          volumes, and plain-language summaries on one page — built for demos
          and thesis presentations.
        </p>
        <div className="mx-auto mt-8 max-w-2xl rounded-2xl border border-disclaimer-border/40 bg-disclaimer/80 px-4 py-3 text-left text-sm text-disclaimer-foreground backdrop-blur-md">
          {RESEARCH_DISCLAIMER}
        </div>
      </motion.div>
    </section>
  );
}
