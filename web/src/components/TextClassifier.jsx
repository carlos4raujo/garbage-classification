import * as tf from "@tensorflow/tfjs"
import { Button, Form, FormLabel } from "react-bootstrap"

import labels from "../constants/labels"

import vocab from "../assets/word_index.json"

const TextClassifier = ({ show, model, setPredict, setIsLoadingPredict }) => {
  if (!show) return null

  const preprocess = (text) => {
    setIsLoadingPredict(true)
    setPredict(null)
    const maxLen = model.inputs[0].shape[1]
    const words = text.toLowerCase().split(/\s+/)
    const seq = words.map((w) => vocab[w] || 0)
    const padded = new Array(maxLen).fill(0)
    for (let i = 0; i < Math.min(seq.length, maxLen); i++) padded[i] = seq[i]
    return tf.tensor2d([padded])
  }

  const predictText = async (text) => {
    const input = preprocess(text)
    const data = await model.predict(input).data()
    const index = data.indexOf(Math.max(...data))
    // setTimeout(() => {
      setIsLoadingPredict(false)
      setPredict(labels[index])
    // }, 4000)
  }

  const handleSubmit = (event) => {
    event.preventDefault()
    const formData = new FormData(event.target)
    const textInput = formData.get("textInput")
    predictText(textInput)
  }

  return (
    <Form onSubmit={handleSubmit}>
      <FormLabel>Describe el residuo</FormLabel>
      <Form.Control type="text" placeholder="Ingresa un objeto" name="textInput" />
      <Button
        size="sm"
        variant="success"
        style={{ width: "100%", rowGap: 12 }}
        type="submit"
        className="d-flex justify-content-center my-4 px-4"
      >
        <span>Clasificar residuo</span>
      </Button>
    </Form>
  )
}

export default TextClassifier
