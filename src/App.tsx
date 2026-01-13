import { useEffect, useRef, useState } from 'react'
import './App.css'
import { SchematicRenderer } from 'schematic-renderer'

type Status = 'initializing' | 'idle' | 'loading' | 'ready' | 'error'

type LoadingProgress = {
  stage: 'file_reading' | 'parsing' | 'mesh_building' | 'scene_setup'
  progress: number
  message: string
}

const PACK_URL = './pack.zip'

const nameFromUrl = (url: string) => {
  try {
    const last = new URL(url).pathname.split('/').filter(Boolean).pop()
    return last || 'schematic'
  } catch {
    return url
  }
}

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const rendererRef = useRef<SchematicRenderer | null>(null)
  const urlRef = useRef<string | null>(null)
  const urlErrorRef = useRef(false)

  const [status, setStatus] = useState<Status>('initializing')
  const [message, setMessage] = useState<string>('Preparing renderer...')
  const [progress, setProgress] = useState<number | null>(null)
  const [schematicName, setSchematicName] = useState<string>('')
  const [schematicUrl, setSchematicUrl] = useState<string | null>(null)
  const [rendererReady, setRendererReady] = useState(false)
  const [canvasSize, setCanvasSize] = useState(() => ({
    width: typeof window !== 'undefined' ? window.innerWidth : 1920,
    height: typeof window !== 'undefined' ? window.innerHeight : 1080,
  }))

  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    const rawUrl = params.get('url')

    if (!rawUrl) {
      setMessage('Add ?url=<schematic url> to render automatically or drop a file.')
      return
    }

    try {
      const resolvedUrl = new URL(rawUrl, window.location.href).toString()
      urlRef.current = resolvedUrl
      setSchematicUrl(resolvedUrl)
    } catch {
      urlErrorRef.current = true
      setStatus('error')
      setMessage('The ?url parameter is not a valid URL.')
    }
  }, [])

  useEffect(() => {
    urlRef.current = schematicUrl
  }, [schematicUrl])

  useEffect(() => {
    const handleResize = () =>
      setCanvasSize({ width: window.innerWidth, height: window.innerHeight })

    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  const [initializeRenderer, setInitializeRenderer] = useState(false);
  useEffect(() => {
    // Delay initialization to ensure canvasRef is set
    const timeout = setTimeout(() => setInitializeRenderer(true), 10);
    return () => clearTimeout(timeout);
  }, []);

  useEffect(() => {
    if (!initializeRenderer || !canvasRef.current) return

    let disposed = false
    let packBlob: Blob | null = null
    const canvasElement = canvasRef.current

    const initialiseRenderer = async () => {
      console.log('Initializing SchematicRenderer...')
      setStatus((current) => (current === 'error' ? current : 'initializing'))
      setMessage((current) => current || 'Preparing renderer...')

      try {
        const response = await fetch(PACK_URL)
        if (!response.ok) {
          throw new Error(`pack.zip not found at ${PACK_URL} (${response.status})`)
        }

        packBlob = await response.blob()

        const renderer = new SchematicRenderer(
          canvasElement,
          {},
          {
            vanillaPack: async () => {
              if (!packBlob) {
                throw new Error('Resource pack has not finished loading.')
              }
              return packBlob
            },
          },
          {
            singleSchematicMode: true,
            enableDragAndDrop: true,
            enableAutoOrbit: false,
            autoOrbitDuration: 28,
            enableProgressBar: true,
            cameraOptions: { defaultCameraPreset: 'isometric' },
            enableInteraction: true,
            gizmoOptions: {
              enableRotation: true,
              enableScaling: true,
            },
            callbacks: {
              onRendererInitialized: () => {
                if (disposed) return
                setRendererReady(true)
                setStatus((current) => {
                  if (current === 'error') return current
                  return urlRef.current ? 'loading' : 'idle'
                })

                if (!urlRef.current) {
                  setMessage('Drop a schematic or pass ?url=<...> to load one.')
                }
              },
              onSchematicLoaded: (name) => {
                if (disposed) return
                setSchematicName(name)
                setStatus('ready')
                setMessage('')
                setProgress(null)
              },
              onSchematicDropSuccess: (file) => {
                if (disposed) return
                setSchematicName(file.name)
                setStatus('ready')
                setMessage('')
                setProgress(null)
              },
              onSchematicDropFailed: (file, error) => {
                if (disposed) return
                setStatus('error')
                setMessage(error?.message || `Failed to load ${file.name}`)
              },
              onInvalidFileType: (file) => {
                if (disposed) return
                setStatus('error')
                setMessage(`Unsupported file type: ${file.name}`)
              },
            },
          },
        )

        rendererRef.current = renderer
      } catch (error) {
        if (disposed) return
        console.error('Failed to initialize SchematicRenderer', error)
        setStatus('error')
        setMessage(error instanceof Error ? error.message : 'Failed to initialize renderer.')
      }
    }

    initialiseRenderer()

    return () => {
      disposed = true
      rendererRef.current?.dispose?.()
      rendererRef.current = null
      setRendererReady(false)
      console.log('Renderer disposed')
    }
  }, [initializeRenderer])

  useEffect(() => {
    if (!rendererReady || !schematicUrl || urlErrorRef.current) {
      return
    }

    let cancelled = false
    let waitTimeout: number | undefined

    const waitForSchematicManager = () =>
      new Promise<NonNullable<SchematicRenderer['schematicManager']>>((resolve, reject) => {
        const start = Date.now()

        const check = () => {
          if (cancelled) return

          const manager = rendererRef.current?.schematicManager
          if (manager) {
            resolve(manager)
            return
          }

          if (Date.now() - start > 5000) {
            reject(new Error('Schematic manager did not initialize in time.'))
            return
          }

          waitTimeout = window.setTimeout(check, 100)
        }

        check()
      })

    const loadSchematicFromUrl = async () => {
      setStatus('loading')
      setMessage('Fetching schematic...')
      setProgress(0)

      try {
        const schematicManager = await waitForSchematicManager()
        if (cancelled) return

        const readableName = nameFromUrl(schematicUrl)

        await schematicManager.removeAllSchematics()
        await schematicManager.loadSchematicFromURL(
          schematicUrl,
          readableName,
          { focused: true },
          {
            onProgress: (loadingProgress: LoadingProgress) => {
              if (cancelled) return
              setProgress(Math.round(loadingProgress.progress))
              if (loadingProgress.message) {
                setMessage(loadingProgress.message)
              }
            },
          },
        )

        if (cancelled) return
        setSchematicName(readableName)
        setStatus('ready')
        setMessage('')
        setProgress(null)
        //rendererRef.current?.setAutoOrbit(true)
      } catch (error) {
        if (cancelled) return
        console.error('Unable to load schematic', error)
        setStatus('error')
        setMessage(error instanceof Error ? error.message : 'Failed to load schematic.')
      }
    }

    loadSchematicFromUrl()

    return () => {
      cancelled = true
      if (waitTimeout !== undefined) {
        window.clearTimeout(waitTimeout)
      }
    }
  }, [rendererReady, schematicUrl])

  const statusLabel: Record<Status, string> = {
    initializing: 'Initializing renderer',
    idle: 'Waiting for a schematic',
    loading: 'Loading schematic',
    ready: 'Renderer ready',
    error: 'Something went wrong',
  }

  const resolvedMessage =
    message ||
    (status === 'ready' && schematicName ? `Showing ${schematicName}` : 'Waiting for a schematic...')

  return (
    <div className="app-shell">
      <canvas
        ref={canvasRef}
        className="renderer-canvas"
        width={canvasSize.width}
        height={canvasSize.height}
      />

      <div className="status-bar">
        <span className={`status-dot status-${status}`} />
        <div className="status-text">
          <div className="status-label">{statusLabel[status]}</div>
          <div className="status-subtext">{resolvedMessage}</div>
          {progress !== null && (
            <div className="progress">
              <div className="progress-bar" style={{ width: `${Math.min(progress, 100)}%` }} />
            </div>
          )}
        </div>
      </div>

      {status === 'idle' && (
        <div className="helper-card">
          <h2>Send me a schematic</h2>
          <p>Append <code>?url=</code> to this page or drop a .schem/.litematic file anywhere.</p>
          <code className="example-url">
            http://localhost:5173/?url=https://example.com/build.schem
          </code>
        </div>
      )}

      {status === 'error' && (
        <div className="helper-card error">
          <h2>Something went wrong</h2>
          <p>{resolvedMessage}</p>
        </div>
      )}
    </div>
  )
}
