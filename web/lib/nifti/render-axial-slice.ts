/**
 * Decode a NIfTI (.nii / .nii.gz) and render one axial slice to a data URL.
 * Brain-window style display for demo previews (not a radiology workstation).
 */

const BRAIN_MIN = 0;
const BRAIN_MAX = 80;

type NiftiHeader = {
  dims: number[];
  datatypeCode: number;
  numBitsPerVoxel: number;
};

type NiftiModule = {
  isCompressed: (data: ArrayBuffer) => boolean;
  decompress: (data: ArrayBuffer) => ArrayBuffer;
  isNIFTI: (data: ArrayBuffer) => boolean;
  readHeader: (data: ArrayBuffer) => NiftiHeader;
  readImage: (header: NiftiHeader, data: ArrayBuffer) => ArrayBuffer;
};

function typedArrayFor(
  header: NiftiHeader,
  buffer: ArrayBuffer,
): Float32Array {
  const code = header.datatypeCode;
  let raw: ArrayLike<number>;
  if (code === 2) raw = new Uint8Array(buffer);
  else if (code === 4) raw = new Int16Array(buffer);
  else if (code === 8) raw = new Int32Array(buffer);
  else if (code === 16) raw = new Float32Array(buffer);
  else if (code === 64) raw = new Float64Array(buffer);
  else if (code === 512) raw = new Uint16Array(buffer);
  else raw = new Float32Array(buffer);

  const out = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i++) out[i] = Number(raw[i]);
  return out;
}

export type AxialPreview = {
  dataUrl: string;
  width: number;
  height: number;
  sliceIndex: number;
  sliceCount: number;
};

export async function renderAxialPreview(
  file: File,
  sliceRatio = 0.5,
): Promise<AxialPreview> {
  const nifti = (await import("nifti-reader-js")) as unknown as NiftiModule;

  let data: ArrayBuffer = await file.arrayBuffer();
  if (nifti.isCompressed(data)) {
    data = nifti.decompress(data);
  }
  if (!nifti.isNIFTI(data)) {
    throw new Error("File is not a valid NIfTI volume.");
  }

  const header = nifti.readHeader(data);
  const image = nifti.readImage(header, data);
  const values = typedArrayFor(header, image);

  const nx = header.dims[1];
  const ny = header.dims[2];
  const nz = header.dims[3] || 1;
  const sliceIndex = Math.min(
    nz - 1,
    Math.max(0, Math.floor((nz - 1) * sliceRatio)),
  );

  const canvas = document.createElement("canvas");
  canvas.width = nx;
  canvas.height = ny;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas unavailable");

  const imageData = ctx.createImageData(nx, ny);
  const scale = BRAIN_MAX - BRAIN_MIN;
  const plane = nx * ny;
  const offset = sliceIndex * plane;

  for (let y = 0; y < ny; y++) {
    for (let x = 0; x < nx; x++) {
      const srcY = ny - 1 - y;
      const v = values[offset + srcY * nx + x];
      const t = Math.max(0, Math.min(1, (v - BRAIN_MIN) / scale));
      const g = Math.round(t * 255);
      const i = (y * nx + x) * 4;
      imageData.data[i] = g;
      imageData.data[i + 1] = g;
      imageData.data[i + 2] = g;
      imageData.data[i + 3] = 255;
    }
  }

  ctx.putImageData(imageData, 0, 0);
  return {
    dataUrl: canvas.toDataURL("image/png"),
    width: nx,
    height: ny,
    sliceIndex,
    sliceCount: nz,
  };
}
