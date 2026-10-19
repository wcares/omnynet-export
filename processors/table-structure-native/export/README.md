# Exporting the TableFormerV2 graphs

Copied from the spike (2026-10-17) that proved the export; they are the record of how the three graphs were made, not a tool. Each script has `S = "..."`
at the top: set it to a work directory first. They need `torch`, `transformers`, `huggingface_hub`, `onnxscript` and `docling-ibm-models` (4.0.3) at EXPORT time
only; the processor needs none of them.

- `export_tf.py` writes the image encoder (`tf_encoder.onnx`, classic exporter, opset 17) and also prints the PyTorch reference tokens.
- `export_dec2.py` writes the token decoder over the whole prefix, no KV cache (`tf_decoder_dynamo.onnx`, `dynamo=True`, dynamic sequence length).
- `export_box2.py` writes the cell box head (`tf_bbox_dynamo.onnx`, `dynamo=True`, dynamic cell count).

The processor reads them as `encoder.onnx`, `decoder.onnx` and `bbox.onnx` in its model directory: rename `tf_encoder.onnx`, `tf_decoder_dynamo.onnx`, `tf_bbox_dynamo.onnx`
accordingly. The classic tracer bakes the sequence length into a Reshape for the decoder and box head, which is why those two use the dynamo exporter.
