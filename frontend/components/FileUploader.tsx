"use client"

import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"

type Props = {
  onUpload: (files: File[]) => void
  disabled?: boolean
  multiple?: boolean
  mode?: "classify" | "ingest"  // ← new
}

export default function FileUploader({ onUpload, disabled, multiple = true, mode = "classify" }: Props) {
  const [dragOver, setDragOver] = useState(false)
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setDragOver(false)
    const files = Array.from(e.dataTransfer.files || [])
    if (files.length > 0) onUpload(files)
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    if (files.length > 0) onUpload(files)
  }

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault()
        setDragOver(true)
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition ${
           dragOver
             ? "border-blue-500 bg-blue-100 dark:border-blue-400 dark:bg-blue-900"
             : "border-gray-300 bg-white dark:border-gray-600 dark:bg-gray-800"
        }`}
      onClick={() => fileInputRef.current?.click()}
    >
      <p className="text-sm text-gray-500 dark:text-gray-300 mb-1">
        Drag and drop file(s) here or click to select
      </p>
      <p  className="text-xs text-gray-400 dark:text-gray-500 mb-3">
        {mode === "ingest"
          ? "📚 These files will be added to the knowledge base."
          : "🧠 These files will be extracted and classified."}
      </p>
      <Button type="button" disabled={disabled} variant="secondary">
        Browse Files
      </Button>
      <input
        type="file"
        accept=".txt,.pdf,.docx"
        multiple={multiple}
        hidden
        ref={fileInputRef}
        onChange={handleFileSelect}
      />
    </div>
  )
}
