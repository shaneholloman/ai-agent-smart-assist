"use client"

import { useTheme } from "next-themes"
import { Textarea } from "@/components/ui/textarea"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import LoadingSpinner from "@/components/LoadingSpinner"
import { useEffect, useState } from "react"

type UploadItem = {
  filename: string
  mode: "classify" | "ingest"
}

type Props = {
  path: string
  setPath: (val: string) => void
  loading: boolean
  handleRunPipeline: () => void
  uploadHistory: UploadItem[]
  onHistoryClick: (filename: string) => void
  mode: "classify" | "ingest"
}

export default function Sidebar({
  path,
  setPath,
  loading,
  handleRunPipeline,
  uploadHistory,
  onHistoryClick,
  mode,
}: Props) {
  const { theme, setTheme } = useTheme()
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  return (
    <aside className="w-64 bg-gray-900 text-white dark:bg-gray-900 p-6 flex flex-col space-y-6">
      <div className="text-xl font-bold">LangChain Agent</div>

      {/* Mode Display */}
      <div className="text-xs text-gray-400 mt-1">
        Mode:{" "}
        <span className="font-semibold text-white">
          {mode === "classify" ? "🧠 Classify & Route" : "📚 Ingest to KB"}
        </span>
      </div>

      {/* Pipeline Section */}
      <div className="space-y-2">
        <h2 className="text-sm font-semibold uppercase tracking-wide">Pipeline</h2>
        <Textarea
          placeholder="Enter a directory path"
          value={path}
          onChange={(e) => setPath(e.target.value)}
          className="bg-white text-black dark:bg-gray-800 dark:text-white placeholder-gray-500 dark:placeholder-gray-400"
        />
        <Button
          disabled={loading || !path}
          onClick={handleRunPipeline}
          className="w-full"
        >
          {loading ? (
            <div className="flex items-center gap-2 justify-center">
              <LoadingSpinner />
              Running...
            </div>
          ) : (
            "Run Pipeline"
          )}
        </Button>
      </div>

      {/* Upload History */}
      <div className="space-y-2">
        <h2 className="text-sm font-semibold uppercase tracking-wide">Upload History</h2>
        <ScrollArea className="h-48 pr-2">
          <ul className="space-y-2 text-sm">
            {uploadHistory.length === 0 ? (
              <li className="text-gray-400">No uploads yet</li>
            ) : (
              uploadHistory.map((item, idx) => (
                <li
                  key={idx}
                  className="bg-gray-800 px-3 py-2 rounded hover:bg-gray-700 cursor-pointer flex justify-between items-center"
                  onClick={() => onHistoryClick(item.filename)}
                >
                  <span className="truncate">{item.filename}</span>
                  <span className="text-xs text-gray-400">
                    {item.mode === "ingest" ? "📚" : "🧠"}
                  </span>
                </li>
              ))
            )}
          </ul>
        </ScrollArea>
      </div>

      {/* Theme Toggle */}
      <div className="pt-6 border-t border-gray-700 mt-auto">
        <Button
          variant="secondary"
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          className="w-full text-sm"
        >
          Toggle {mounted ? (theme === "dark" ? "Light" : "Dark") : "Theme"} Mode
        </Button>
      </div>
    </aside>
  )
}
