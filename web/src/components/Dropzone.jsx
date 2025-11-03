import { useCallback } from "react"

import { CloudUpload } from "lucide-react"
import { useDropzone } from "react-dropzone"

const Dropzone = () => {
  const onDrop = useCallback((acceptedFiles) => {
    console.log(acceptedFiles)
  }, [])

  const { getRootProps, getInputProps } = useDropzone({ onDrop })

  return (
    <div {...getRootProps()}>
      <input {...getInputProps()} />
      <div className="dropzone d-flex flex-column align-items-center justify-content-center">
        <CloudUpload size={36} />
        <span className="fs-5 fw-semibold">Haz click para subir o arrastra una imagen</span>
        <span className="fs-6 opacity-75">PNG, JPG, WEBP hasta 10MB</span>
      </div>
    </div>
  )
}

export default Dropzone
