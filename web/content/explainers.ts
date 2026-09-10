/** Beginner-friendly copy for tooltips and accordions. No clinical advice. */

export const METRIC_TOOLTIPS = {
  dice:
    "Measures how closely the AI prediction matches the expert annotation. Higher values indicate better segmentation.",
  iou: "Measures overlap between predicted and actual hemorrhage.",
  confidence:
    "Shows how certain the AI is about this prediction. This is a research score — not a clinical probability.",
  volume_error:
    "Difference between predicted and actual hemorrhage volume. Lower is better.",
  inference_time: "Time required for the AI model to analyze the scan.",
  model_size: "How large the trained model file is on disk.",
  recommended_use: "When this project recommends using this model.",
  volume: "Estimated amount of hemorrhage tissue the model found in the scan.",
  processing_time: "Total time from start of analysis to final summary.",
} as const;

export const LEARN_ACCORDION = [
  {
    id: "hemorrhage-types",
    title: "Hemorrhage types",
    body: `Intracerebral hemorrhage appears in several common patterns on CT. This project looks for subtypes such as intraparenchymal (IPH), subdural (SDH), epidural (EDH), intraventricular (IVH), and subarachnoid (SAH). The AI draws a region on the scan for each type it thinks is present. Labels come from research datasets — they are not a radiologist’s final report.`,
  },
  {
    id: "dice",
    title: "What is Dice?",
    body: `Dice is a score from 0 to 1 that asks: “How much do the AI’s drawn region and the expert’s drawn region agree?” 1 means perfect match. 0 means no overlap. On this page, comparison Dice values come from locked test benchmarks for each model — not from your uploaded file unless a reference label was provided.`,
  },
  {
    id: "iou",
    title: "What is IoU?",
    body: `IoU (Intersection over Union) is another overlap score. It divides shared area by the combined area of prediction and truth. Like Dice, higher is better. We show it so you can see agreement another way, without needing machine-learning training.`,
  },
  {
    id: "confidence",
    title: "What does confidence mean?",
    body: `Confidence summarizes how sure the model appears about its own output on this scan. High confidence does not guarantee correctness. Low confidence means the result should be treated with extra caution. Never use this number alone for medical decisions.`,
  },
  {
    id: "volume",
    title: "What is volume?",
    body: `Volume is the estimated size of the highlighted hemorrhage region, usually in milliliters (mL). It comes from counting predicted voxels and converting with the scan’s spacing. Small differences between models are common.`,
  },
  {
    id: "model-comparison",
    title: "How to read model comparison",
    body: `The table places four models side by side. MONAI is the fast interactive demo (locked-test Dice 0.257). nnU-Net is the V1 3D comparator (0.455, lowest volume error 14.3 mL). nnU-Net Ensemble averages two folds (locked-test Dice 0.489, IoU 0.343). SSL means semi-supervised learning: one nnU-Net trained on labeled CTs plus teacher masks (Dice 0.471). IoU and volume error were not scored for SSL.`,
  },
  {
    id: "monai-vs-nnunet",
    title: "MONAI vs nnU-Net vs nnU-Net Ensemble vs SSL",
    body: `MONAI: fastest CPU feedback, locked-test Dice 0.257. nnU-Net: slower, Dice 0.455, best volume error. nnU-Net Ensemble: two folds averaged, locked-test Dice 0.489. SSL (semi-supervised learning): labeled + unlabeled training, Dice 0.471. Compare runs your chosen models one after another on the same file.`,
  },
] as const;

export type ComparisonWinner = "monai" | "nnunet" | "ssl" | "semi" | "tie";

export const COMPARISON_ROWS = [
  {
    id: "dice",
    metric: "Dice",
    monai: "0.257",
    nnunet: "0.455",
    ssl: "0.489",
    semi: "0.471",
    meaning: "Overlap with expert labels (locked-test Dice)",
    tooltip: METRIC_TOOLTIPS.dice,
    winner: "ssl" as ComparisonWinner,
  },
  {
    id: "iou",
    metric: "IoU",
    monai: "0.163",
    nnunet: "0.313",
    ssl: "0.343",
    semi: "—",
    meaning: "Another way to measure region overlap",
    tooltip: METRIC_TOOLTIPS.iou,
    winner: "ssl" as ComparisonWinner,
  },
  {
    id: "volume_error",
    metric: "Volume error",
    monai: "19.5 mL mean",
    nnunet: "14.3 mL mean",
    ssl: "18.5 mL mean",
    semi: "—",
    meaning: "How far predicted volume sits from expert volume",
    tooltip: METRIC_TOOLTIPS.volume_error,
    winner: "nnunet" as ComparisonWinner,
  },
  {
    id: "confidence",
    metric: "Confidence",
    monai: "Study score",
    nnunet: "Study score",
    ssl: "Study score",
    semi: "Study score",
    meaning: "Model self-certainty on a run (research only)",
    tooltip: METRIC_TOOLTIPS.confidence,
    winner: "tie" as ComparisonWinner,
  },
  {
    id: "inference_time",
    metric: "Inference time",
    monai: "~20 s (CPU)",
    nnunet: "~5+ min (CPU)",
    ssl: "~2× nnU-Net",
    semi: "~5+ min (CPU)",
    meaning: "Typical wait to finish one scan on CPU",
    tooltip: METRIC_TOOLTIPS.inference_time,
    winner: "monai" as ComparisonWinner,
  },
  {
    id: "model_size",
    metric: "Model size",
    monai: "Compact",
    nnunet: "1 fold",
    ssl: "2 folds",
    semi: "1 fold",
    meaning: "Relative footprint of weights on disk",
    tooltip: METRIC_TOOLTIPS.model_size,
    winner: "monai" as ComparisonWinner,
  },
  {
    id: "recommended_use",
    metric: "Recommended use",
    monai: "Interactive demos",
    nnunet: "V1 research",
    ssl: "nnU-Net Ensemble",
    semi: "SSL",
    meaning: "Best fit inside this project",
    tooltip: METRIC_TOOLTIPS.recommended_use,
    winner: "tie" as ComparisonWinner,
  },
] as const;
