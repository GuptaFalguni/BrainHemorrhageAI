export const APP_NAME = "BrainHemorrhageAI";
export const APP_TAGLINE =
  "AI-powered Brain Hemorrhage Segmentation and Volume Analysis";

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ||
  "http://127.0.0.1:8000";

export const RESEARCH_DISCLAIMER =
  "Research Software — Not for Clinical Use. This tool performs investigational CT hemorrhage segmentation. It is not a medical device and must not be used for diagnosis, triage, or treatment decisions.";
