import { InferenceContainer } from '@/components/inference-container'

export default function Home() {
  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold mb-2">YOLOv9 Object Detection</h1>
            <p className="text-muted-foreground">
              Upload an image to detect objects using YOLOv9 AI model
            </p>
          </div>

          <InferenceContainer />
        </div>
      </div>
    </div>
  )
}
