import { API_BASE_URL } from "@/lib/constants";
import { ApiError } from "@/lib/api/client";
import type { PredictResponse } from "@/types/predict";

export async function postPredict(
  file: File,
  modelId: string,
): Promise<PredictResponse> {
  const url = `${API_BASE_URL}/api/v1/predict`;
  const form = new FormData();
  form.append("file", file);
  form.append("model_id", modelId);

  let response: Response;
  try {
    response = await fetch(url, {
      method: "POST",
      body: form,
      cache: "no-store",
    });
  } catch {
    throw new ApiError(`Cannot reach API at ${API_BASE_URL}`, 0);
  }

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: string };
      if (typeof body.detail === "string") detail = body.detail;
      else if (body.detail) detail = JSON.stringify(body.detail);
    } catch {
      /* ignore */
    }
    throw new ApiError(detail || `Predict failed (${response.status})`, response.status);
  }

  return (await response.json()) as PredictResponse;
}

function artifactUrl(downloadPath: string | undefined): string | null {
  if (!downloadPath) return null;
  if (downloadPath.startsWith("http")) return downloadPath;
  return `${API_BASE_URL}${downloadPath.startsWith("/") ? "" : "/"}${downloadPath}`;
}

function pickOverlayUrl(downloadUrls: Record<string, string>): string | null {
  return (
    artifactUrl(downloadUrls["overlay.png"]) ||
    artifactUrl(downloadUrls["prediction_overlay.png"])
  );
}

function pickOverlay3dUrl(downloadUrls: Record<string, string>): string | null {
  return artifactUrl(downloadUrls["prediction_overlay_3d.png"]);
}

/** Fetch API overlay into a same-origin blob URL (avoids blank cross-origin img issues). */
export async function fetchOverlayBlobUrl(
  downloadUrls: Record<string, string>,
): Promise<{ overlayUrl: string | null; overlay3dUrl: string | null }> {
  const overlayPath = pickOverlayUrl(downloadUrls);
  const overlay3dPath = pickOverlay3dUrl(downloadUrls);

  async function toBlobUrl(url: string | null): Promise<string | null> {
    if (!url) return null;
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Overlay download failed (${response.status})`);
    }
    const blob = await response.blob();
    if (!blob.type.startsWith("image/") && blob.size < 1000) {
      throw new Error("Overlay response was not an image");
    }
    return URL.createObjectURL(blob);
  }

  const [overlayUrl, overlay3dUrl] = await Promise.all([
    toBlobUrl(overlayPath),
    toBlobUrl(overlay3dPath),
  ]);
  return { overlayUrl, overlay3dUrl };
}

