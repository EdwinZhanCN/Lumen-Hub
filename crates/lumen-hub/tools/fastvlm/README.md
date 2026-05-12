# FastVLM Assets

Expected layout:

```text
model_info.json
tokenizer.json
onnx/vision.fp32.onnx
onnx/embed.fp32.onnx
onnx/decoder.fp32.onnx
```

The first `lumen-hub` task, `vlm_embeds`, uses `vision` and `embed`.
`decoder` is declared in the package contract for later prefill/decode tasks.
