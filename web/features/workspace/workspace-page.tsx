"use client";

import { AnalyzeSection } from "@/components/workspace/analyze-section";
import { ClinicalSummarySection } from "@/components/workspace/clinical-summary-section";
import { ComparisonSection } from "@/components/workspace/comparison-section";
import { DisclaimerSection } from "@/components/workspace/disclaimer-section";
import { DownloadsSection } from "@/components/workspace/downloads-section";
import { HeroSection } from "@/components/workspace/hero-section";
import { LearnSection } from "@/components/workspace/learn-section";
import { ModelSelectSection } from "@/components/workspace/model-select-section";
import { ProgressSection } from "@/components/workspace/progress-section";
import { ResultsSection } from "@/components/workspace/results-section";
import { UploadSection } from "@/components/workspace/upload-section";

export function WorkspacePage() {
  return (
    <div className="relative min-h-screen">
      <div
        aria-hidden
        className="pointer-events-none fixed inset-0 -z-10"
        style={{
          backgroundImage:
            "radial-gradient(circle at 20% 20%, color-mix(in oklab, var(--primary) 8%, transparent), transparent 40%), radial-gradient(circle at 80% 0%, color-mix(in oklab, var(--primary) 6%, transparent), transparent 35%)",
        }}
      />
      <HeroSection />
      <UploadSection />
      <ModelSelectSection />
      <AnalyzeSection />
      <ProgressSection />
      <ResultsSection />
      <ComparisonSection />
      <ClinicalSummarySection />
      <DownloadsSection />
      <LearnSection />
      <DisclaimerSection />
    </div>
  );
}
