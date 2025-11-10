import { useCallback, useEffect, useState } from "react"

import { CloudUpload } from "lucide-react"
import { useDropzone } from "react-dropzone"
import { Button } from "react-bootstrap"

const ImageClassifier = ({ show }) => {
  const [selectedFile, setSelectedFile] = useState()
  const [preview, setPreview] = useState()

  const onDrop = useCallback((acceptedFiles) => {
    console.log(acceptedFiles)
    if (!acceptedFiles || acceptedFiles.length === 0) {
      setSelectedFile(undefined)
      return
    }

    setSelectedFile(acceptedFiles[0])
  }, [])

  const { getRootProps, getInputProps } = useDropzone({ onDrop })

  useEffect(() => {
    if (!selectedFile) {
      setPreview(undefined)
      return
    }

    const objectUrl = URL.createObjectURL(selectedFile)
    setPreview(objectUrl)

    return () => URL.revokeObjectURL(objectUrl)
  }, [selectedFile])

  const onClearImage = () => {
    setSelectedFile(undefined)
  }

  if (!show) return null

  return (
    <>
      {preview ? (
        <>
          <div
            style={{
              width: 400,
              height: 400,
              margin: "0px auto 16px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              overflow: "hidden",
              borderRadius: 8,
              backgroundImage: `url(${preview})`,
              backgroundSize: "cover",
            }}
          />
          <Button size="sm" color="info" className="my-4 mx-auto d-block" onClick={onClearImage}>
            Limpiar
          </Button>
        </>
      ) : (
        <div {...getRootProps()}>
          <input {...getInputProps()} />

          <div className="dropzone d-flex flex-column align-items-center justify-content-center">
            <CloudUpload size={36} />
            <span className="fs-5 fw-semibold">Haz click para subir o arrastra una imagen</span>
            <span className="fs-6 opacity-75">PNG, JPG, WEBP hasta 10MB</span>
          </div>
        </div>
      )}
      {preview && <Button
        size="sm"
        variant="success"
        style={{ width: "100%", rowGap: 12 }}
        type="submit"
        className="d-flex justify-content-center my-4 px-4"
      >
        <span>Clasificar residuo</span>
      </Button>}
    </>
  )
}

export default ImageClassifier
