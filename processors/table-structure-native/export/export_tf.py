import torch, time, json, numpy as np, torchvision.transforms as T
from PIL import Image
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from docling_ibm_models.tableformer_v2.model import TableFormerV2
p = snapshot_download("docling-project/TableFormerV2")
m = TableFormerV2.from_pretrained(p).eval(); tok = AutoTokenizer.from_pretrained(p)
print("bos", tok.bos_token_id, "eos", tok.eos_token_id, "vocab", len(tok), flush=True)
tf = T.Compose([T.Resize((448, 448)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
x = tf(Image.open(S + "/crop0.png").convert("RGB")).unsqueeze(0) if (S := "/data/worktrees/vbacklog/rig-artifacts/docling-spike/table") else None
t = time.time()
with torch.no_grad(): ref = m.generate(x, tok, max_length=512)
print("torch generate", round(time.time() - t, 2), "s", ref["generated_ids"].shape, flush=True)
ids = ref["generated_ids"][0].tolist(); print("tokens:", [tok.decode([i]) for i in ids][:40]); print("bboxes", None if ref["predicted_bboxes"] is None else ref["predicted_bboxes"].shape)

class Enc(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, images): return s.m.encode_images(images)["last_hidden_state"]
class Dec(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, input_ids, memory):
        L = input_ids.shape[1]
        tgt = s.m.input_embedding(input_ids) + s.m._positional_encoding(1, L, 0)
        mask = torch.triu(torch.ones(L, L)).T
        mask = mask.masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)
        decoded, _ = s.m.transformer_decoder(tgt=tgt, memory=memory, tgt_mask=mask, tgt_key_padding_mask=None, past_key_values=None, use_cache=False)
        return s.m.output_projection(decoded), decoded
class Box(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, cell_embs, memory):
        return s.m.bbox_head(cell_embs, memory, torch.zeros(cell_embs.shape[0], dtype=torch.long), spatial_size=(14, 14))
enc, dec, box = Enc(m).eval(), Dec(m).eval(), Box(m).eval()
with torch.no_grad(): mem = enc(x)
print("memory", mem.shape, flush=True)
def exp(mod, args, name, inputs, outputs, dyn):
    t = time.time()
    try:
        torch.onnx.export(mod, args, S + "/" + name, input_names=inputs, output_names=outputs, dynamic_axes=dyn, opset_version=17, dynamo=False)
        print("EXPORT OK", name, round(time.time() - t, 1), "s", flush=True)
    except Exception as e:
        print("EXPORT FAILED", name, str(e)[:300], flush=True)
exp(enc, (x,), "tf_encoder.onnx", ["images"], ["memory"], None)
exp(dec, (torch.tensor([ids[:5]]), mem), "tf_decoder.onnx", ["input_ids", "memory"], ["logits", "hidden"], {"input_ids": {1: "L"}, "logits": {1: "L"}, "hidden": {1: "L"}})
exp(box, (torch.randn(7, 512), mem), "tf_bbox.onnx", ["cell_embs", "memory"], ["boxes"], {"cell_embs": {0: "N"}, "boxes": {0: "N"}})
torch.save({"x": x, "ids": ref["generated_ids"], "boxes": ref["predicted_bboxes"]}, S + "/ref.pt")
