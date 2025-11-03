import { useState } from "react"
import { FileCheck } from 'lucide-react'
import { Card, Container, Button, Row, Col, Form, FormLabel } from "react-bootstrap"

import Header from "./components/Header"
import Dropzone from "./components/Dropzone"

function App() {
  const [type, setType] = useState("image")

  return (
    <div style={{ backgroundColor: 'oklch(.98 .005 120)', height: '100vh' }}>
      <Header />
      <Container>
        <Card className="shadow-sm mt-3 mx-auto" style={{ maxWidth: 650 }}>
          <div className="p-3">
            <Row className="mb-4">
              <Col md="6">
                <Button variant="success" size="sm" style={{ width: "100%" }} onClick={() => setType("image")}>
                  Subir imagen
                </Button>
              </Col>
              <Col md="6">
                <Button style={{ width: "100%" }} size="sm" variant="success" onClick={() => setType("text")}>
                  Describir texto
                </Button>
              </Col>
            </Row>
            {type === "image" ? (
              <Dropzone />
            ) : (
              <Form>
                <FormLabel>Describe el residuo</FormLabel>
                <Form.Control type="text" placeholder="Ingresa un objeto" />
              </Form>
            )}
          </div>
          <Row className="my-4 px-4">
            <Col md="12">
              <Button size="sm" variant="success" style={{ width: "100%", rowGap: 12 }} onClick={() => setType("image")} className="d-flex justify-content-center">
                <span>Clasificar residuo</span>
              </Button>
            </Col>
          </Row>
        </Card>
      </Container>
    </div>
  )
}

export default App
