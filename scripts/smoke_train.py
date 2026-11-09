#!/usr/bin/env python3
"""Run one real training epoch on a short pair list and verify optimization."""
import argparse, json, math, os, sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'examples/WanVSR/ablation_experiments'))
sys.path.insert(0,str(root))
p=argparse.ArgumentParser(add_help=False)
p.add_argument('--smoke-stage',choices=['main','decoder'],required=True)
a,remaining=p.parse_known_args()
sys.argv=[sys.argv[0],*remaining]
if a.smoke_stage=='main':
    import train_plan_atlas_loop_ddp as entry
    trainer_class=entry.AtlasLoopTrainerDDP
else:
    import train_plan_atlas_loop_tcdecoder_lora_mri_ddp as entry
    trainer_class=entry.TcdecoderLoraMriTrainerDDP
base=entry.base_train
import torch
args=entry.get_args()
base.validate_gpu_layout(args)
base.setup_distributed()
trainer=None
try:
    trainer=trainer_class(args)
    before=[p.detach().float().sum().item() for p in trainer._lora_params]
    loss=trainer.train_epoch_stage3(0)
    after=[p.detach().float().sum().item() for p in trainer._lora_params]
    if not math.isfinite(loss) or trainer.global_step<1: raise RuntimeError('No finite optimizer step completed')
    if before==after: raise RuntimeError('No trainable parameter changed')
    if not all(torch.isfinite(p).all() for p in trainer._lora_params): raise RuntimeError('Non-finite parameters')
    if base.is_dist():
        checksum=torch.tensor(after,device=trainer.device,dtype=torch.float64)
        reference=checksum.clone()
        torch.distributed.broadcast(reference,src=0)
        torch.testing.assert_close(checksum,reference,rtol=1e-5,atol=1e-4)
    if base.is_main(): trainer.save_checkpoint(0,0.0,0.0)
    base.barrier()
    trainer.val_loader.dataset.full_volume=False
    metrics=trainer.validate(0,save_images=False)
    if not all(math.isfinite(float(x)) for x in metrics[:4]): raise RuntimeError('Non-finite validation metrics')
    from diffsynth.models.wan_video_dit import BLOCK_SPARSE_ATTN_AVAILABLE
    result={'stage':a.smoke_stage,'world_size':base.get_world_size(),'steps_per_rank':trainer.global_step,
            'attention_backend':'block_sparse_attn' if BLOCK_SPARSE_ATTN_AVAILABLE else 'block_masked_pytorch_sdpa',
            'loss':float(loss),'metrics':list(metrics[:4]),'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
            'checkpoint':str(Path(trainer.output_dir)/'checkpoints/epoch_1.pth')}
    if base.is_main():
        (Path(args.output_dir)/'smoke_result.json').write_text(json.dumps(result,indent=2)+'\n')
        print('SMOKE_PASSED '+json.dumps(result),flush=True)
finally:
    if trainer is not None: trainer.pipe.shutdown_vllm()
    base.cleanup_distributed()
