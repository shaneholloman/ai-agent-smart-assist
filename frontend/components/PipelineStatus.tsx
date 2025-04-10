import { Card } from "@/components/ui/card"

export default function PipelineStatus({ pipelineStatus }: { pipelineStatus: any }) {
  return (
    <div className="space-y-2">
      <h2 className="text-xl font-semibold">📂 Pipeline Output</h2>
      <Card>
        <pre className="bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-white p-4 rounded-md text-sm overflow-x-auto whitespace-pre-wrap border border-gray-300 dark:border-gray-700">
          {typeof pipelineStatus === "string"
            ? pipelineStatus
            : JSON.stringify(pipelineStatus, null, 2)}
        </pre>
      </Card>
    </div>
  )
}
