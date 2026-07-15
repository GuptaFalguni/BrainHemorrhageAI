import Link from "next/link";

import { EmptyState } from "@/components/feedback/empty-state";
import { PageContainer } from "@/components/layout/page-container";
import { Button } from "@/components/ui/button";

export default function NotFound() {
  return (
    <PageContainer>
      <EmptyState
        title="Page not found"
        description="That route does not exist in the BrainHemorrhageAI workbench."
      />
      <div className="mt-4 flex justify-center">
        <Button asChild>
          <Link href="/">Back to home</Link>
        </Button>
      </div>
    </PageContainer>
  );
}
