# 1. 创建 model root 和 upstream/source.json
mkdir -p /tmp/mobileclip2-s4/upstream
cat > /tmp/mobileclip2-s4/upstream/source.json << 'EOF'
{
"format": "openclip",
"repo_id": "timm/MobileCLIP2-S4-OpenCLIP"
}
EOF

# 2. 运行导出
python3 crates/lumen-hub/tools/clip/scripts/export_clip_to_onnx.py --model-root /tmp/mobileclip2-s4
