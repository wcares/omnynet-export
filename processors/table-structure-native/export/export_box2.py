import torch, time
from huggingface_hub import snapshot_download
from docling_ibm_models.tableformer_v2.model import TableFormerV2
S = "/data/worktrees/vbacklog/rig-artifacts/docling-spike/table"
m = TableFormerV2.from_pretrained(snapshot_download("docling-project/TableFormerV2")).eval()
class Box(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, cell_embs, memory):
        return s.m.bbox_head(cell_embs, memory, torch.zeros(cell_embs.shape[0], dtype=torch.long), spatial_size=(14, 14))
box = Box(m).eval()
N = torch.export.Dim("N", min=1, max=1023)
t = time.time()
prog = torch.onnx.export(box, (torch.randn(7, 512), torch.randn(1, 196, 512)), dynamic_shapes={"cell_embs": {0: N}, "memory": None}, dynamo=True)
prog.save(S + "/tf_bbox_dynamo.onnx"); print("BBOX DYNAMO OK", round(time.time() - t, 1))
