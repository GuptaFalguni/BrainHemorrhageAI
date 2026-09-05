"use client";

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { LEARN_ACCORDION } from "@/content/explainers";

export function LearnSection() {
  return (
    <section id="learn" className="scroll-mt-20 px-4 py-10 sm:px-6">
      <div className="mx-auto max-w-3xl">
        <h2 className="mb-2 text-center text-2xl font-semibold tracking-tight">
          Understanding Your Results
        </h2>
        <p className="mb-8 text-center text-sm text-muted-foreground">
          Beginner-friendly explanations — no ML jargon required.
        </p>

        <div className="glass-strong rounded-3xl px-5 sm:px-8">
          <Accordion type="single" collapsible className="w-full">
            {LEARN_ACCORDION.map((item) => (
              <AccordionItem key={item.id} value={item.id}>
                <AccordionTrigger>{item.title}</AccordionTrigger>
                <AccordionContent>{item.body}</AccordionContent>
              </AccordionItem>
            ))}
          </Accordion>
        </div>
      </div>
    </section>
  );
}
