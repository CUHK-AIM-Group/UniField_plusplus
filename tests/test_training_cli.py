import subprocess,sys,unittest
from pathlib import Path

class CliTests(unittest.TestCase):
    def test_decoder_preserves_device_and_loop_arguments(self):
        root=Path(__file__).resolve().parents[1]
        code="""
import sys
sys.path.insert(0,'examples/WanVSR/ablation_experiments')
sys.argv=['decoder','--config','configs/train.yaml','--gpu','7,3,1',
          '--vlm_device','cuda:2','--loop_iterations','3','--loop_block_start','5',
          '--loop_block_end','7','--source_checkpoint','example.pth']
import train_plan_atlas_loop_tcdecoder_lora_mri_ddp as entry
a=entry.get_args()
assert (a.gpu,a.vlm_device,a.loop_iterations,a.loop_block_start,a.loop_block_end)==('7,3,1','cuda:2',3,5,7)
"""
        result=subprocess.run([sys.executable,'-c',code],cwd=root,capture_output=True,text=True,timeout=90)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

if __name__=='__main__': unittest.main()
