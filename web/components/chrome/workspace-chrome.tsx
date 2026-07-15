"use client";

import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";

import { ThemeToggle } from "@/components/ui/theme-toggle";
import { getHealth } from "@/lib/api";
import { APP_NAME } from "@/lib/constants";
import { queryKeys } from "@/lib/query-keys";
import { cn } from "@/lib/utils";

export function WorkspaceChrome() {
  const health = useQuery({
    queryKey: queryKeys.health,
    queryFn: getHealth,
    refetchInterval: 30_000,
    retry: 1,
  });

  const online = health.isSuccess && health.data.status === "ok";
  const device = health.data?.device ?? "—";

  return (
    <header className="sticky top-0 z-40 border-b border-glass-border bg-background/70 backdrop-blur-xl">
      <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-4 sm:px-6">
        <div className="flex items-center gap-3">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="flex size-8 items-center justify-center rounded-xl bg-primary/15 text-xs font-semibold text-primary"
            aria-hidden
          >
            BH
          </motion.div>
          <span className="text-sm font-semibold tracking-tight sm:text-base">
            {APP_NAME}
          </span>
        </div>

        <div className="flex items-center gap-2 sm:gap-3">
          <div
            className={cn(
              "inline-flex items-center gap-2 rounded-full border px-2.5 py-1 text-xs",
              online
                ? "border-success/30 bg-success/10 text-success"
                : "border-border bg-surface-muted text-muted-foreground",
            )}
            role="status"
            aria-live="polite"
          >
            <span
              className={cn(
                "size-1.5 rounded-full",
                online ? "bg-success" : "bg-muted-foreground",
              )}
            />
            <span className="hidden sm:inline">
              {online ? `API OK · ${device}` : health.isLoading ? "Checking…" : "API offline"}
            </span>
            <span className="sm:hidden">{online ? "OK" : "…"}</span>
          </div>
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}
