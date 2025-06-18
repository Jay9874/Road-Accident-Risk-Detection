// frontend/src/App.jsx
import React, { useRef, useEffect, useState } from 'react'

function App () {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const wsRef = useRef(null)
  const [status, setStatus] = useState('Connecting to backend...')
  const [alertTriggered, setAlertTriggered] = useState(false)

  useEffect(() => {
    const video = videoRef.current
    const canvas = canvasRef.current
    const context = canvas.getContext('2d')

    let animationFrameId

    const connectWebSocket = () => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.close()
      }

      wsRef.current = new WebSocket('ws://localhost:8000/ws/drowsiness') // Adjust port if your backend runs on a different one

      wsRef.current.onopen = () => {
        console.log('WebSocket connection opened.')
        setStatus('Webcam access requested...')
        navigator.mediaDevices
          .getUserMedia({ video: true })
          .then(stream => {
            video.srcObject = stream
            video.play()
            setAlertTriggered(false) // Reset alert status on new connection/play
            // Start sending frames once video metadata is loaded
            video.onloadedmetadata = () => {
              canvas.width = video.videoWidth
              canvas.height = video.videoHeight
              sendFrame()
            }
          })
          .catch(err => {
            console.error('Error accessing webcam:', err)
            setStatus(
              'Error: Could not access webcam. Please allow camera access.'
            )
          })
      }

      wsRef.current.onmessage = event => {
        const data = JSON.parse(event.data)
        // console.log("Received data:", data);

        // Redraw current video frame on canvas
        context.clearRect(0, 0, canvas.width, canvas.height)
        context.drawImage(video, 0, 0, canvas.width, canvas.height)

        setStatus(
          `Status: ${data.status} | Drowsy Frames: ${data.drowsy_frame_count}`
        )
        if (data.alert_triggered) {
          setAlertTriggered(true)
        }

        // Draw face box
        data.face_boxes.forEach(box => {
          // Ensure color conversion (BGR from Python to RGB for CSS)
          const [r, g, b] = data.status_color.reverse() // Reverse for RGB
          context.strokeStyle = `rgb(${r}, ${g}, ${b})`
          context.lineWidth = 2
          context.strokeRect(box[0], box[1], box[2] - box[0], box[3] - box[1])
        })

        // Draw mouth boxes
        data.mouth_boxes.forEach(box => {
          const [x1, y1, x2, y2, label, confidence] = box
          context.strokeStyle = label === 'Yawn' ? 'red' : 'green'
          context.lineWidth = 1
          context.strokeRect(x1, y1, x2 - x1, y2 - y1)
          context.fillStyle = label === 'Yawn' ? 'red' : 'green'
          context.font = '12px Arial'
          context.fillText(
            `<span class="math-inline">${label} (</span>{confidence})`,
            x1,
            y1 - 5
          )
        })

        // Draw eye boxes
        data.eye_boxes.forEach(box => {
          const [x1, y1, x2, y2, label, confidence] = box
          context.strokeStyle = label === 'Closed' ? 'red' : 'yellow'
          context.lineWidth = 1
          context.strokeRect(x1, y1, x2 - x1, y2 - y1)
          context.fillStyle = label === 'Closed' ? 'red' : 'yellow'
          context.font = '12px Arial'
          context.fillText(
            `<span class="math-inline">${label} (</span>{confidence})`,
            x1,
            y1 - 5
          )
        })
      }

      wsRef.current.onclose = event => {
        console.log('WebSocket connection closed:', event)
        setStatus('Disconnected. Reconnecting in 3 seconds...')
        setTimeout(connectWebSocket, 3000) // Attempt to reconnect
      }

      wsRef.current.onerror = event => {
        console.error('WebSocket error:', event)
        setStatus('WebSocket error. Check console. Reconnecting...')
        wsRef.current.close() // Force close to trigger reconnect
      }
    }

    const sendFrame = () => {
      if (
        video.readyState === video.HAVE_ENOUGH_DATA &&
        wsRef.current &&
        wsRef.current.readyState === WebSocket.OPEN
      ) {
        context.drawImage(video, 0, 0, canvas.width, canvas.height)
        canvas.toBlob(
          blob => {
            if (blob) {
              const reader = new FileReader()
              reader.onload = () => {
                const base64data = reader.result.split(',')[1]
                wsRef.current.send(base64data)
              }
              reader.readAsDataURL(blob)
            }
          },
          'image/jpeg',
          0.7
        ) // Adjust quality here (0.7 is a good balance)
      }
      animationFrameId = requestAnimationFrame(sendFrame)
    }

    connectWebSocket()

    // Cleanup on component unmount
    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
      if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop())
      }
      cancelAnimationFrame(animationFrameId)
    }
  }, []) // Empty dependency array means this effect runs once on mount

  return (
    <div style={{ textAlign: 'center', fontFamily: 'Arial, sans-serif' }}>
      <h1>Live Drowsiness Detection</h1>
      <div style={{ position: 'relative', width: '640px', margin: 'auto' }}>
        {/* The video element is hidden but used to grab frames */}
        <video
          ref={videoRef}
          style={{ display: 'none' }}
          playsInline
          muted
        ></video>
        {/* The canvas displays the video feed with overlays */}
        <canvas
          ref={canvasRef}
          style={{ border: '2px solid #ccc', borderRadius: '8px' }}
        ></canvas>
      </div>
      <p style={{ fontSize: '1.2em', fontWeight: 'bold', marginTop: '10px' }}>
        {status}
      </p>
      {alertTriggered && (
        <p style={{ color: 'red', fontSize: '1.5em', fontWeight: 'bold' }}>
          DROWSINESS ALERT SENT!
        </p>
      )}
      <p>
        This application uses your webcam feed to detect drowsiness. Your video
        feed is processed on the backend, and results are displayed here.
      </p>
    </div>
  )
}

export default App
