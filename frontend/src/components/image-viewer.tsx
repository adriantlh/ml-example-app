'use client'

import { useState, useRef, useEffect } from 'react'
import Image from 'next/image'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

interface Detection {
  class_id: number
  class_name: string
  confidence: number
  bbox: [number, number, number, number] // [x1, y1, x2, y2]
}

interface InferenceResult {
  filename: string
  detections: Detection[]
  count: number
}

interface ImageViewerProps {
  imageUrl: string
  result: InferenceResult
}

// Color palette for different classes
const COLORS = [
  '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57',
  '#FF9FF3', '#54A0FF', '#5F27CD', '#00D2D3', '#FF9F43',
  '#FC5C65', '#26DE81', '#2BCBBA', '#EB3B5A', '#F7B731'
]

export function ImageViewer({ imageUrl, result }: ImageViewerProps) {
  const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0 })
  const [displayDimensions, setDisplayDimensions] = useState({ width: 0, height: 0 })
  const imageRef = useRef<HTMLImageElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const img = new window.Image()
    img.onload = () => {
      setImageDimensions({ width: img.width, height: img.height })
    }
    img.src = imageUrl
  }, [imageUrl])

  useEffect(() => {
    const updateDisplayDimensions = () => {
      if (imageRef.current && containerRef.current) {
        const rect = imageRef.current.getBoundingClientRect()
        setDisplayDimensions({ width: rect.width, height: rect.height })
      }
    }

    updateDisplayDimensions()
    window.addEventListener('resize', updateDisplayDimensions)

    return () => {
      window.removeEventListener('resize', updateDisplayDimensions)
    }
  }, [imageUrl])

  const scaleCoordinates = (bbox: [number, number, number, number]) => {
    if (!imageDimensions.width || !imageDimensions.height || !displayDimensions.width || !displayDimensions.height) {
      return { x: 0, y: 0, width: 0, height: 0 }
    }

    const [x1, y1, x2, y2] = bbox
    const scaleX = displayDimensions.width / imageDimensions.width
    const scaleY = displayDimensions.height / imageDimensions.height

    return {
      x: x1 * scaleX,
      y: y1 * scaleY,
      width: (x2 - x1) * scaleX,
      height: (y2 - y1) * scaleY
    }
  }

  const getColorForClass = (classId: number) => {
    return COLORS[classId % COLORS.length]
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Detection Results</CardTitle>
          <p className="text-sm text-muted-foreground">
            Found {result.count} object{result.count !== 1 ? 's' : ''} in {result.filename}
          </p>
        </CardHeader>
        <CardContent>
          <div ref={containerRef} className="relative inline-block">
            <Image
              ref={imageRef}
              src={imageUrl}
              alt="Analyzed image"
              width={800}
              height={600}
              className="max-w-full h-auto rounded-lg border"
              unoptimized={true}
              onLoad={() => {
                if (imageRef.current && containerRef.current) {
                  const rect = imageRef.current.getBoundingClientRect()
                  setDisplayDimensions({ width: rect.width, height: rect.height })
                }
              }}
            />

            {/* Bounding boxes overlay */}
            {result.detections.map((detection, index) => {
              const coords = scaleCoordinates(detection.bbox)
              const color = getColorForClass(detection.class_id)

              return (
                <div
                  key={index}
                  className="absolute pointer-events-none"
                  style={{
                    left: coords.x,
                    top: coords.y,
                    width: coords.width,
                    height: coords.height,
                    border: `2px solid ${color}`,
                    backgroundColor: `${color}20`,
                  }}
                >
                  {/* Label */}
                  <div
                    className="absolute -top-6 left-0 px-1 py-0.5 text-xs font-medium text-white rounded text-nowrap"
                    style={{ backgroundColor: color }}
                  >
                    {detection.class_name} ({(detection.confidence * 100).toFixed(1)}%)
                  </div>
                </div>
              )
            })}
          </div>
        </CardContent>
      </Card>

      {/* Detection details */}
      <Card>
        <CardHeader>
          <CardTitle>Detection Details</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {result.detections.map((detection, index) => (
              <div
                key={index}
                className="flex items-center justify-between p-2 rounded-lg border"
              >
                <div className="flex items-center space-x-2">
                  <div
                    className="w-4 h-4 rounded"
                    style={{ backgroundColor: getColorForClass(detection.class_id) }}
                  />
                  <span className="font-medium">{detection.class_name}</span>
                </div>
                <div className="text-sm text-muted-foreground">
                  {(detection.confidence * 100).toFixed(1)}% confidence
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}