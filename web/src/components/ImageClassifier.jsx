import { useCallback, useEffect, useState } from "react"

import { CloudUpload } from "lucide-react"
import { useDropzone } from "react-dropzone"

const ImageClassifier = ({ show }) => {
  const [selectedFile, setSelectedFile] = useState()
  const [preview, setPreview] = useState()
  const onDrop = useCallback((acceptedFiles) => {
    console.log(acceptedFiles)
    if (!acceptedFiles || acceptedFiles.length === 0) {
      setSelectedFile(undefined)
      return
    }
    
    // I've kept this example simple by using the first image instead of multiple
    setSelectedFile(acceptedFiles[0])
  }, [])
  
  const { getRootProps, getInputProps } = useDropzone({ onDrop })
  
  // create a preview as a side effect, whenever selected file is changed
  useEffect(() => {
    if (!selectedFile) {
      setPreview(undefined)
      return
    }

    const objectUrl = URL.createObjectURL(selectedFile)
    setPreview(objectUrl)

    // free memory when ever this component is unmounted
    return () => URL.revokeObjectURL(objectUrl)
  }, [selectedFile])

  if (!show) return null

  return (
    <div {...getRootProps()}>
      <input {...getInputProps()} />
      {selectedFile &&  <img src={preview} /> }
      <div className="dropzone d-flex flex-column align-items-center justify-content-center">
        <CloudUpload size={36} />
        <span className="fs-5 fw-semibold">Haz click para subir o arrastra una imagen</span>
        <span className="fs-6 opacity-75">PNG, JPG, WEBP hasta 10MB</span>
      </div>
    </div>
  )
}

export default ImageClassifier
