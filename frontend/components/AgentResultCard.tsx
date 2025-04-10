"use client"

import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"

type Props = {
  task: string
  output: Record<string, any>
  trace?: Record<string, any>
}

export default function AgentResultCard({ task, output, trace }: Props) {
  return (
    <Card className="space-y-4 mt-6">
      <div className="flex items-center justify-between">
        <Badge>Task: {task}</Badge>
        {trace?.routed_tool && <Badge>Tool: {trace.routed_tool}</Badge>}
      </div>

      <ScrollArea className="max-h-[400px] pr-4">
        <pre className="text-sm whitespace-pre-wrap">
          {JSON.stringify(output, null, 2)}
        </pre>
      </ScrollArea>
    </Card>
  )
}
