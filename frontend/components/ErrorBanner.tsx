"use client"

type Props = {
  message: string
}

export default function ErrorBanner({ message }: Props) {
  return (
    <div className="text-red-600 text-sm mb-4 bg-red-100 border border-red-300 p-3 rounded">
      ⚠️ {message}
    </div>
  )
}
