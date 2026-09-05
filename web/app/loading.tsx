import { LoadingCard } from "@/components/feedback/loading-card";
import { PageContainer } from "@/components/layout/page-container";

export default function Loading() {
  return (
    <PageContainer>
      <div className="mb-4 h-8 w-48 rounded shimmer" />
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        <LoadingCard />
        <LoadingCard />
        <LoadingCard />
      </div>
    </PageContainer>
  );
}
