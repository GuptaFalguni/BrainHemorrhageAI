"use client";

import { useEffect } from "react";

import { ErrorState } from "@/components/feedback/error-state";
import { PageContainer } from "@/components/layout/page-container";

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error(error);
  }, [error]);

  return (
    <PageContainer>
      <ErrorState
        title="Something went wrong"
        detail={error.message || "Unexpected application error."}
        onRetry={reset}
      />
    </PageContainer>
  );
}
