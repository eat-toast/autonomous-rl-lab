"""Apply approved pacing feedback while preserving the original pilot."""
from pathlib import Path
import sys, subprocess, json, hashlib
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'.tools'))
sys.path.insert(0,str(ROOT))
import imageio_ffmpeg
from scripts.commit_review import validate_review, commit_review

validate_review(ROOT/'studies/MLP-01/results')
commit = commit_review(ROOT/'studies/MLP-01/results')
base=ROOT/'studies/MLP-01/exports/instagram_MLP-01_EP02'
source=base/'pixel_pilot_v1/pixel_maze_pilot_v1.mp4'
out=base/'pixel_pilot_v2'
out.mkdir(exist_ok=True)
video=out/'pixel_maze_pilot_v2.mp4'
ff=imageio_ffmpeg.get_ffmpeg_exe()
# Original frames 90..419: complete 100-action playback + one-second result.
subprocess.run([ff,'-y','-v','error','-i',str(source),'-vf','trim=start_frame=90:end_frame=420,setpts=PTS-STARTPTS','-an','-c:v','libx264','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True)
subprocess.run([ff,'-v','error','-i',str(video),'-f','null','-'],check=True,capture_output=True)
for label,frame in [('start',0),('arrival',90),('result',300),('end',329)]:
    subprocess.run([ff,'-y','-v','error','-i',str(video),'-vf',f'select=eq(n\\,{frame})','-frames:v','1',str(out/f'{label}.png')],check=True)
record=json.loads((base/'pixel_pilot_v1/production.json').read_text('utf-8'))
record.update(status='local_revision_ready',review_commit=commit,seconds=11,frames=330,source_video_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),edit='Keep original frames [90,420): remove 3-second static intro and final 4 seconds. All recorded actions retained; result held 1 second.')
(out/'production.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(video)
