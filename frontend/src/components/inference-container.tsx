'use client'

import { useState } from 'react'
import { ImageUpload } from '@/components/image-upload'
import { ImageViewer } from '@/components/image-viewer'

interface Detection {
  class_id: number
  class_name: string
  confidence: number
  bbox: [number, number, number, number]
}

interface InferenceResult {
  filename: string
  detections: Detection[]
  count: number
}

export function InferenceContainer() {
  const [inferenceResult, setInferenceResult] = useState<InferenceResult | null>(null)
  const [imageUrl, setImageUrl] = useState<string | null>(null)

  const handleInferenceComplete = (result: InferenceResult, url: string) => {
    setInferenceResult(result)
    setImageUrl(url)
  }

  const handleReset = () => {
    setInferenceResult(null)
    setImageUrl(null)
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
      <div>
        <h2 className="text-xl font-semibold mb-4">Upload Image</h2>
        <ImageUpload onInferenceComplete={handleInferenceComplete} />

        {inferenceResult && (
          <div className="mt-4">
            <button
              onClick={handleReset}
              className="text-sm text-muted-foreground hover:text-foreground underline"
            >
              Upload another image
            </button>
          </div>
        )}
      </div>

      <div>
        <h2 className="text-xl font-semibold mb-4">Results</h2>
        {inferenceResult && imageUrl ? (
          <ImageViewer imageUrl={imageUrl} result={inferenceResult} />
        ) : (
          <div className="border-2 border-dashed border-border rounded-lg p-8 text-center text-muted-foreground">
            Upload an image to see detection results
          </div>
        )}
      </div>
    </div>
  )
}