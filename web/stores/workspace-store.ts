"use client";

import { create } from "zustand";

export type ModelMode = "monai" | "nnunet" | "compare";

export type RunPhase =
  | "idle"
  | "ready"
  | "running"
  | "complete"
  | "error";

export type TimelineStepId =
  | "upload"
  | "preprocessing"
  | "segmentation"
  | "volume"
  | "clinical"
  | "completed";

export type SubtypeVolume = {
  code: string;
  fullName: string;
  volumeMl: number;
  percent: number;
  isPrimary: boolean;
};

export type ModelResult = {
  modelId: string;
  displayName: string;
  predictionId?: string;
  subtype: string;
  subtypeFull: string;
  volumeMl: number;
  confidence: number;
  inferenceSeconds: number;
  processingSeconds: number;
  hemorrhageDetected?: boolean;
  subtypeVolumes: SubtypeVolume[];
  overlaySrc: string;
  overlay3dSrc?: string | null;
  originalSrc?: string | null;
};

type WorkspaceState = {
  file: File | null;
  fileName: string | null;
  fileSize: number | null;
  mode: ModelMode;
  phase: RunPhase;
  timelineStep: TimelineStepId;
  elapsedMs: number;
  errorMessage: string | null;
  results: ModelResult[];
  originalPreviewUrl: string | null;
  progressLabel: string | null;
  setMode: (mode: ModelMode) => void;
  setFile: (file: File | null) => void;
  setPhase: (phase: RunPhase) => void;
  setTimelineStep: (step: TimelineStepId) => void;
  setElapsedMs: (ms: number) => void;
  setError: (message: string | null) => void;
  setResults: (results: ModelResult[]) => void;
  clearResults: () => void;
  setOriginalPreviewUrl: (url: string | null) => void;
  setProgressLabel: (label: string | null) => void;
};

export const TIMELINE_STEPS: { id: TimelineStepId; label: string }[] = [
  { id: "upload", label: "Upload" },
  { id: "preprocessing", label: "Preprocessing" },
  { id: "segmentation", label: "Segmentation" },
  { id: "volume", label: "Volume Analysis" },
  { id: "clinical", label: "Clinical Summary" },
  { id: "completed", label: "Completed" },
];

export const useWorkspaceStore = create<WorkspaceState>((set) => ({
  file: null,
  fileName: null,
  fileSize: null,
  mode: "monai",
  phase: "idle",
  timelineStep: "upload",
  elapsedMs: 0,
  errorMessage: null,
  results: [],
  originalPreviewUrl: null,
  progressLabel: null,
  setMode: (mode) => set({ mode }),
  setFile: (file) =>
    set({
      file,
      fileName: file?.name ?? null,
      fileSize: file?.size ?? null,
      phase: file ? "ready" : "idle",
      results: [],
      errorMessage: null,
      timelineStep: "upload",
      elapsedMs: 0,
      originalPreviewUrl: null,
      progressLabel: null,
    }),
  setPhase: (phase) => set({ phase }),
  setTimelineStep: (timelineStep) => set({ timelineStep }),
  setElapsedMs: (elapsedMs) => set({ elapsedMs }),
  setError: (errorMessage) =>
    set({ errorMessage, phase: errorMessage ? "error" : "idle" }),
  setResults: (results) => set({ results, phase: "complete" }),
  clearResults: () => set({ results: [] }),
  setOriginalPreviewUrl: (originalPreviewUrl) => set({ originalPreviewUrl }),
  setProgressLabel: (progressLabel) => set({ progressLabel }),
}));
