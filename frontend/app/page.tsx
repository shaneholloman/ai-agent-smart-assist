"use client"

import { useState } from "react"
import { Textarea } from "@/components/ui/textarea"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"
import Sidebar from "@/components/Sidebar"
import AgentResultCard from "@/components/AgentResultCard"
import PipelineStatus from "@/components/PipelineStatus"
import ErrorBanner from "@/components/ErrorBanner"
import FileUploader from "@/components/FileUploader"
import LoadingSpinner from "@/components/LoadingSpinner"
import "@/app/globals.css"
import { useRef, useEffect } from "react"

type AgentResponse = {
  task: string
  output: Record<string, any>
  agent_trace?: Record<string, any>
}

type UploadItem = {
  filename: string
  mode: "classify" | "ingest"
}

export default function HomePage() {
  const [text, setText] = useState("")
  const [path, setPath] = useState("")
  const [loading, setLoading] = useState(false)
  const [pipelineStatus, setPipelineStatus] = useState<string | null>(null)
  const [result, setResult] = useState<AgentResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [uploadHistory, setUploadHistory] = useState<UploadItem[]>([])
  const [mode, setMode] = useState<"classify" | "ingest">("classify")
  const [question, setQuestion] = useState("")
  const [queryResults, setQueryResults] = useState<string[]>([])
  const threadIdRef = useRef<string | null>(null)

  const handleClearAll = () => {
    setText("")
    setQuestion("")
    setResult(null)
    setError(null)
    setPipelineStatus(null)
    setQueryResults([])
  }

  useEffect(() => {
      // Load thread_id from localStorage or create a new one
      let savedThreadId = localStorage.getItem("rag_thread_id")
      if (!savedThreadId) {
        savedThreadId = crypto.randomUUID()
        localStorage.setItem("rag_thread_id", savedThreadId)
      }
      threadIdRef.current = savedThreadId
    }, [])

  const handleRunAgent = async () => {
    setError(null)
    setLoading(true)
    setResult(null)

    try {
      const res = await fetch("http://localhost:8000/run-agent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      })

      if (!res.ok) {
        const { detail } = await res.json()
        throw new Error(detail || "Agent error")
      }

      const data = await res.json()
      setResult(data)
    } catch (err: any) {
      setError(err.message || "Something went wrong")
    } finally {
      setLoading(false)
    }
  }

  const handleRunPipeline = async () => {
    setError(null)
    setPipelineStatus(null)
    setLoading(true)

    try {
      const endpoint =
        mode === "ingest"
          ? `http://localhost:8000/run-ingestion-pipeline?path=${encodeURIComponent(path)}&namespace=default`
          : "http://localhost:8000/run-pipeline"

      const res = await fetch(endpoint, {
        method: "POST"
      })

      if (!res.ok) {
        const { detail } = await res.json()
        throw new Error(detail || "Pipeline failed")
      }

      const result = await res.json()
      setPipelineStatus(JSON.stringify(result, null, 2))
    } catch (err: any) {
      setError("Failed to run document pipeline")
    } finally {
      setLoading(false)
    }
  }

  const handleQuery = async () => {
    setError(null)
    setQueryResults([])
    setLoading(true)

    try {
      const threadId = threadIdRef.current
      const res = await fetch(
        `http://localhost:8000/api/query?question=${encodeURIComponent(question)}&namespace=default&thread_id=${threadId}`
      )

      if (!res.ok) {
        const { detail } = await res.json()
        throw new Error(detail || "Query failed")
      }

      const data = await res.json()
      setQueryResults(data.results)
    } catch (err: any) {
      setError(err.message || "Failed to query knowledge base")
    } finally {
      setLoading(false)
    }
  }

  const handleExport = () => {
    if (!result) return
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `agent-output-${result.task}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  const handleHistoryClick = (filename: string) => {
    console.log("Clicked on previous upload:", filename)
  }

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar
        path={path}
        setPath={setPath}
        loading={loading}
        handleRunPipeline={handleRunPipeline}
        uploadHistory={uploadHistory}
        onHistoryClick={handleHistoryClick}
        mode={mode}
      />

      <main className="flex-1 p-8 overflow-y-auto bg-gray-50">
        <h1 className="text-2xl font-semibold mb-6">🧠 Document Classifier Playground</h1>

        {/* Mode selector + Clear */}
        <div className="mb-6 flex items-center justify-between">
          <div>
            <label className="block mb-2 font-medium text-sm text-gray-700">Choose Mode</label>
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value as "classify" | "ingest")}
              className="p-2 border rounded w-64"
            >
              <option value="classify">🧠 Classify & Route</option>
              <option value="ingest">📚 Ingest into Knowledge Base</option>
            </select>
          </div>
          <Button variant="secondary" onClick={handleClearAll}>
            Clear All
          </Button>
        </div>

        {/* File upload section */}
        <div className="mb-8">
          <FileUploader
            disabled={loading}
            multiple={true}
            mode={mode}
            onUpload={async (files: File[]) => {
              setError(null)
              setResult(null)
              setLoading(true)

              const uploadTasks = files.map(async (file) => {
                const filename = file.name
                const formData = new FormData()
                formData.append("files", file)

                const endpoint =
                  mode === "ingest"
                    ? "http://localhost:8000/api/ingest"
                    : "http://localhost:8000/upload-docs"

                try {
                  const res = await fetch(endpoint, {
                    method: "POST",
                    body: formData,
                  })

                  if (!res.ok) {
                    const { detail } = await res.json()
                    throw new Error(detail || "Upload failed.")
                  }

                  if (mode === "classify") {
                    const result = await res.json()
                    const extractedText = result.uploaded?.[0]?.extracted_text || ""
                    return { filename, extractedText }
                  } else if (mode === "ingest") {
                    const result = await res.json()
                    setPipelineStatus(JSON.stringify(result, null, 2))
                    return { filename, extractedText: null }
                  }
                } catch (err: any) {
                  return { filename, error: err.message || "Upload failed" }
                }
              })

              const uploadedResults = await Promise.all(uploadTasks)

              uploadedResults.forEach(({ filename, extractedText, error }) => {
                setUploadHistory((prev) => [
                  { filename, mode },
                  ...prev.filter((f) => f.filename !== filename)
                ])

                if (!error) {
                  if (mode === "classify" && extractedText) {
                    setText((prev) => `${prev}\n\n--- ${filename} ---\n\n${extractedText}`)
                  } else if (mode === "ingest") {
                    setPipelineStatus(`✅ Successfully ingested: ${filename}`)
                  }
                } else {
                  setError(`❌ ${filename}: ${error}`)
                }
              })

              setLoading(false)
            }}
          />
        </div>

        {/* RAG-style question input */}
        {mode === "ingest" && (
          <div className="mb-10">
            <label className="block mb-2 font-medium text-sm text-gray-700">Ask a question from your knowledge base</label>
            <div className="flex gap-2">
              <input
                type="text"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="e.g., What is the status field?"
                className="p-2 border rounded w-full"
              />
              <Button disabled={loading || !question} onClick={handleQuery}>
                Ask
              </Button>
            </div>
          </div>
        )}

        {/* Query Results */}
        {queryResults.length > 0 && (
          <div className="mb-8">
            <h2 className="font-semibold text-lg mb-2">🔍 Retrieved Chunks</h2>
            <ul className="space-y-2 text-sm bg-white p-4 rounded shadow border">
              {queryResults.map((res, i) => (
                <li key={i} className="text-gray-700 leading-relaxed">• {res}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Text input for classification */}
        {mode === "classify" && (
          <div className="space-y-4 mb-10">
            <Textarea
              placeholder="Paste your document chunk here..."
              value={text}
              onChange={(e) => setText(e.target.value)}
            />

            <div className="flex gap-4">
              <Button disabled={loading || !text} onClick={handleRunAgent}>
                {loading ? (
                  <div className="flex items-center gap-2">
                    <LoadingSpinner />
                    Processing...
                  </div>
                ) : (
                  "Run Agent"
                )}
              </Button>
              {result && (
                <>
                  <Button variant="secondary" onClick={() => setResult(null)}>
                    Clear
                  </Button>
                  <Button variant="secondary" onClick={handleExport}>
                    Export JSON
                  </Button>
                </>
              )}
            </div>
          </div>
        )}

        {/* Pipeline output */}
        {pipelineStatus && <PipelineStatus pipelineStatus={pipelineStatus} />}
        {error && <ErrorBanner message={error} />}
        {result && <AgentResultCard task={result.task} output={result.output} trace={result.agent_trace} />}
      </main>
    </div>
  )
}
