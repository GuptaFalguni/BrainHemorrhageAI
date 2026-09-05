import { cn } from "@/lib/utils";

export function LoadingCard({
  className,
  lines = 3,
}: {
  className?: string;
  lines?: number;
}) {
  return (
    <div
      className={cn(
        "rounded-lg border border-border bg-card p-4 shadow-sm",
        className,
      )}
      aria-busy="true"
      aria-label="Loading"
    >
      <div className="mb-3 h-4 w-1/3 rounded shimmer" />
      <div className="space-y-2">
        {Array.from({ length: lines }).map((_, index) => (
          <div
            key={index}
            className="h-3 rounded shimmer"
            style={{ width: `${85 - index * 12}%` }}
          />
        ))}
      </div>
    </div>
  );
}
