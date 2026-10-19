
import torch, time
from huggingface_hub import snapshot_download
from docling_ibm_models.tableformer_v2.model import TableFormerV2
S = "/data/worktrees/vbacklog/rig-artifacts/docling-spike/table"
p = snapshot_download("docling-project/TableFormerV2")
m = TableFormerV2.from_pretrained(p).eval()
class Dec(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, input_ids, memory):
        L = input_ids.shape[1]
        tgt = s.m.input_embedding(input_ids) + s.m._positional_encoding(1, L, 0)
        mask = torch.triu(torch.ones(L, L)).T
        mask = mask.masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)
        decoded, _ = s.m.transformer_decoder(tgt=tgt, memory=memory, tgt_mask=mask, tgt_key_padding_mask=None, past_key_values=None, use_cache=False)
        return s.m.output_projection(decoded), decoded
dec = Dec(m).eval()
ids = torch.tensor([[2, 4, 4, 6, 4, 4, 6, 4]]); mem = torch.randn(1, 196, 512)
L = torch.export.Dim("L", min=2, max=511)
t = time.time()
try:
    prog = torch.onnx.export(dec, (ids, mem), dynamic_shapes={"input_ids": {1: L}, "memory": None}, dynamo=True)
    prog.save(S + "/tf_decoder_dynamo.onnx")
    print("DYNAMO EXPORT OK", round(time.time() - t, 1), "s")
except Exception as e:
    print("DYNAMO EXPORT FAILED", str(e)[:400])
