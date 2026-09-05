"use client";

import { AnimatePresence, motion } from "framer-motion";
import { FileUp, X } from "lucide-react";
import { useCallback, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { useWorkspaceStore } from "@/stores/workspace-store";
import { cn } from "@/lib/utils";

const ACCEPTED = [".nii", ".nii.gz"];

function isNifti(file: File) {
  const name = file.name.toLowerCase();
  return name.endsWith(".nii") || name.endsWith(".nii.gz");
}

export function UploadSection() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileName = useWorkspaceStore((s) => s.fileName);
  const fileSize = useWorkspaceStore((s) => s.fileSize);
  const setFile = useWorkspaceStore((s) => s.setFile);

  const acceptFile = useCallback(
    (file: File | null) => {
      if (!file) {
        setFile(null);
        setError(null);
        return;
      }
      if (!isNifti(file)) {
        setError("Please upload a .nii or .nii.gz file.");
        return;
      }
      setError(null);
      setFile(file);
    },
    [setFile],
  );

  return (
    <section id="upload" className="scroll-mt-20 px-4 py-10 sm:px-6">
      <div className="mx-auto max-w-3xl">
        <h2 className="mb-2 text-center text-2xl font-semibold tracking-tight">
          Upload CT Scan
        </h2>
        <p className="mb-8 text-center text-sm text-muted-foreground">
          Supported formats: .nii · .nii.gz
        </p>

        <motion.div
          layout
          onDragEnter={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            const file = e.dataTransfer.files?.[0] ?? null;
            acceptFile(file);
          }}
          className={cn(
            "glass relative overflow-hidden rounded-3xl p-1 transition-shadow",
            dragging && "ring-2 ring-primary shadow-[0_0_40px_-12px_var(--primary)]",
          )}
        >
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            className={cn(
              "flex w-full flex-col items-center justify-center gap-4 rounded-[1.35rem] px-6 py-16 text-center transition-colors",
              "hover:bg-primary/5 focus-visible:outline-none",
              dragging && "bg-primary/10",
            )}
            aria-label="Upload CT scan"
          >
            <motion.div
              animate={dragging ? { y: -4, scale: 1.05 } : { y: 0, scale: 1 }}
              className="flex size-16 items-center justify-center rounded-2xl bg-primary/15 text-primary"
            >
              <FileUp className="size-7" />
            </motion.div>
            <div>
              <p className="text-base font-medium">Drop your CT scan here</p>
              <p className="mt-1 text-sm text-muted-foreground">
                or click to browse
              </p>
            </div>
          </button>
          <input
            ref={inputRef}
            type="file"
            accept={ACCEPTED.join(",")}
            className="sr-only"
            onChange={(e) => acceptFile(e.target.files?.[0] ?? null)}
          />
        </motion.div>

        <AnimatePresence>
          {error && (
            <motion.p
              initial={{ opacity: 0, y: -4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="mt-3 text-center text-sm text-danger"
              role="alert"
            >
              {error}
            </motion.p>
          )}
        </AnimatePresence>

        <AnimatePresence>
          {fileName && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              className="glass mt-4 flex items-center justify-between gap-3 rounded-2xl px-4 py-3"
            >
              <div className="min-w-0">
                <p className="truncate text-sm font-medium">{fileName}</p>
                {fileSize != null && (
                  <p className="text-xs text-muted-foreground">
                    {(fileSize / (1024 * 1024)).toFixed(2)} MB
                  </p>
                )}
              </div>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                aria-label="Remove file"
                onClick={() => {
                  acceptFile(null);
                  if (inputRef.current) inputRef.current.value = "";
                }}
              >
                <X className="size-4" />
              </Button>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </section>
  );
}
